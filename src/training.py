import logging
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.autograd import grad
from torch.nn import Module
from torch.nn.functional import cross_entropy, softmax
from torch.optim import SGD
from torch.utils.data import DataLoader

from src.data import TestDataset, TrainDataset


Array = Union[np.ndarray, Tensor]


@dataclass
class Trainer:
    inputs: Array
    labels: Array
    is_labelled: np.ndarray
    feat_model_class: Module
    task_model_class: Module
    pool_model_class: Module
    selection_coef: float
    device: str
    config: dict

    @staticmethod
    def set_requires_grad(model: Module, requires_grad: bool = True) -> None:
        for param in model.parameters():
            param.requires_grad = requires_grad

    def train(self, alpha: float, grad_penalty_weight: int, n_epochs: int) -> None:
        self.feat_model = self.feat_model_class().to(self.device)
        self.task_model = self.task_model_class().to(self.device)
        self.pool_model = self.pool_model_class().to(self.device)

        feat_optimizer = SGD(self.feat_model.parameters(), **self.config["optimizer_config"])
        task_optimizer = SGD(self.task_model.parameters(), **self.config["optimizer_config"])
        pool_optimizer = SGD(self.pool_model.parameters(), **self.config["optimizer_config"])

        inds_labelled_train = np.arange(len(self.inputs))[self.is_labelled]
        inds_unlabelled_train = np.arange(len(self.inputs))[~self.is_labelled]

        # Compute the unbalancing ratio, a value between [0, 1], typically 0.1-0.5.
        gamma_ratio = len(inds_labelled_train) / len(inds_unlabelled_train)

        train_loader = DataLoader(
            TrainDataset(
                self.inputs[inds_labelled_train],
                self.labels[inds_labelled_train],
                self.inputs[inds_unlabelled_train],
                self.labels[inds_unlabelled_train],
                transform=self.config["train_transform"],
            ),
            shuffle=True,
            **self.config["train_loader_config"],
        )

        for epoch in range(n_epochs):
            self.feat_model.train()
            self.task_model.train()
            self.pool_model.train()

            n_batch = task_acc = task_loss = 0

            for labelled_inputs, labels, unlabelled_inputs, _ in train_loader:
                labelled_inputs = labelled_inputs.to(self.device)
                labels = labels.to(self.device)
                unlabelled_inputs = unlabelled_inputs.to(self.device)

                # ----------------------------------------------------------------------------------
                # Update feature extractor and task classifier.

                self.set_requires_grad(self.feat_model, requires_grad=True)
                self.set_requires_grad(self.task_model, requires_grad=True)
                self.set_requires_grad(self.pool_model, requires_grad=False)

                labelled_latents = self.feat_model(labelled_inputs)
                unlabelled_latents = self.feat_model(unlabelled_inputs)

                task_predictions = self.task_model(labelled_latents)

                wasserstein_distance = self.compute_wasserstein_distance(
                    unlabelled_latents, labelled_latents, gamma_ratio
                )

                with torch.no_grad():
                    labelled_latents = self.feat_model(labelled_inputs)
                    unlabelled_latents = self.feat_model(unlabelled_inputs)

                gradient_penalty = self.compute_gradient_penalty(
                    unlabelled_latents, labelled_latents
                )

                task_loss = cross_entropy(task_predictions, labels)
                task_loss += alpha * (wasserstein_distance + grad_penalty_weight * gradient_penalty)

                feat_optimizer.zero_grad()
                task_optimizer.zero_grad()
                task_loss.backward()
                feat_optimizer.step()
                task_optimizer.step()

                task_predictions = torch.argmax(task_predictions, dim=1)
                task_acc += torch.sum(task_predictions == labels).item() / len(labels)
                task_loss += task_loss.item()
                n_batch += 1

                # ----------------------------------------------------------------------------------
                # Update pool classifier.

                self.set_requires_grad(self.feat_model, requires_grad=False)
                self.set_requires_grad(self.task_model, requires_grad=False)
                self.set_requires_grad(self.pool_model, requires_grad=True)

                with torch.no_grad():
                    labelled_latents = self.feat_model(labelled_inputs)
                    unlabelled_latents = self.feat_model(unlabelled_inputs)

                wasserstein_distance = self.compute_wasserstein_distance(
                    unlabelled_latents, labelled_latents, gamma_ratio
                )

                gradient_penalty = self.compute_gradient_penalty(
                    unlabelled_latents, labelled_latents
                )

                pool_loss = -alpha * (wasserstein_distance + 2 * gradient_penalty)

                pool_optimizer.zero_grad()
                pool_loss.backward()
                pool_optimizer.step()

            task_acc /= n_batch
            task_loss /= n_batch

            logging.info(f"Epoch {epoch:02}: acc = {task_acc:.4f}, loss = {task_loss:.4f}")

    def compute_wasserstein_distance(
        self, unlabelled_latents: Tensor, labelled_latents: Tensor, gamma_ratio: float
    ) -> Tensor:
        predictions_unlabelled = self.pool_model(unlabelled_latents)
        predictions_labelled = self.pool_model(labelled_latents)

        return torch.mean(predictions_unlabelled) - gamma_ratio * torch.mean(predictions_labelled)

    def compute_gradient_penalty(
        self, unlabelled_latents: Tensor, labelled_latents: Tensor
    ) -> Tensor:
        alpha = torch.rand(len(unlabelled_latents), 1).to(unlabelled_latents.device)

        interpolates = (
            unlabelled_latents + alpha * (labelled_latents - unlabelled_latents),
            unlabelled_latents,
            labelled_latents,
        )
        interpolates = torch.cat(interpolates).requires_grad_()

        predictions = self.pool_model(interpolates)

        gradients = grad(
            predictions,
            interpolates,
            grad_outputs=torch.ones_like(predictions),
            retain_graph=True,
            create_graph=True,
        )

        gradient_norm = torch.norm(gradients[0], p=2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

        return gradient_penalty

    @torch.inference_mode()
    def predict_labels_argmax(self, inputs: Tensor, labels: Tensor) -> Tensor:
        self.feat_model.eval()
        self.task_model.eval()

        loader = self.get_test_loader(inputs, labels)
        all_predictions = []

        for inputs, _ in loader:
            inputs = inputs.to(self.device)
            latents = self.feat_model(inputs)

            predictions = self.task_model(latents)
            predictions = torch.argmax(predictions, dim=1)

            all_predictions.append(predictions.cpu())

        return torch.cat(all_predictions)

    @torch.inference_mode()
    def predict_labels_probs(self, inputs: Tensor, labels: Tensor) -> Tensor:
        self.feat_model.eval()
        self.task_model.eval()

        loader = self.get_test_loader(inputs, labels)
        all_predictions = []

        for inputs, _ in loader:
            inputs = inputs.to(self.device)
            latents = self.feat_model(inputs)

            predictions = self.task_model(latents)
            predictions = softmax(predictions, dim=1)

            all_predictions.append(predictions.cpu())

        return torch.cat(all_predictions)

    @torch.inference_mode()
    def predict_pool_scores(self, inputs: Tensor, labels: Tensor) -> Tensor:
        self.feat_model.eval()
        self.pool_model.eval()

        loader = self.get_test_loader(inputs, labels)
        all_predictions = []

        for inputs, _ in loader:
            inputs = inputs.to(self.device)
            latents = self.feat_model(inputs)

            predictions = self.pool_model(latents).reshape(-1)

            all_predictions.append(predictions.cpu())

        return torch.cat(all_predictions)

    def get_test_loader(self, inputs: Tensor, labels: Tensor) -> DataLoader:
        loader = DataLoader(
            TestDataset(inputs, labels, transform=self.config["test_transform"]),
            shuffle=False,
            **self.config["test_loader_config"],
        )
        return loader

    def single_worst(self, probs: Tensor) -> Tensor:
        worst, _ = torch.max(-torch.log(probs), dim=1)
        return worst

    def l1_upper(self, probs: Tensor) -> Tensor:
        return -torch.sum(torch.log(probs), dim=1)

    def l2_upper(self, probs: Tensor) -> Tensor:
        return torch.norm(torch.log(probs), dim=1)

    def select_queries(self, n_query: int) -> np.ndarray:
        inds_unlabelled = np.arange(len(self.inputs))[~self.is_labelled]

        probs = self.predict_labels_probs(
            self.inputs[inds_unlabelled], self.labels[inds_unlabelled]
        )

        uncertainty_scores = 0.5 * (self.l1_upper(probs) + self.l2_upper(probs))

        pool_scores = self.predict_pool_scores(
            self.inputs[inds_unlabelled], self.labels[inds_unlabelled]
        )

        total_scores = uncertainty_scores - self.selection_coef * pool_scores

        selected_inds = torch.argsort(total_scores)[:n_query]

        return inds_unlabelled[selected_inds]
