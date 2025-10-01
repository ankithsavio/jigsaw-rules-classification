"""
Script for cross validation / evaluation
"""

import multiprocessing as mp
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import torch
from sklearn.metrics import (  # type: ignore
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (  # type: ignore
    StratifiedKFold,
)

from jigsaw_rules.configs import (
    ChatConfig,
    DebertaConfig,
    InstructConfig,
    RobertaConfig,
)
from jigsaw_rules.inference import (
    ChatEngine,
    DebertaEngine,
    InstructEngine,
    RobertaEngine,
)
from jigsaw_rules.train import (
    DebertaBase,
    Instruct,
    RobertaBase,
)
from jigsaw_rules.utils import get_train_dataframe


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class JigsawEval:
    def __init__(
        self,
        data_path,
        model_path,
        lora_path=None,
        save_path=None,
    ):
        self.data_path = data_path
        self.model_path = model_path
        self.lora_path = lora_path if lora_path else ""
        self.save_path = save_path if save_path else ""

    def run(self):
        raise NotImplementedError

    def create_dashboard(
        self,
        cv_results,
        fold_predictions,
        model_name,
        filename="cv_dashboard.png",
    ):
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))

        # Define the grid layout
        gs = plt.GridSpec(3, 3, figure=fig)  # type: ignore

        # Plot 1: CV Metrics across folds (top left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        folds = list(range(1, len(cv_results) + 1))
        metrics = ["f1", "precision", "recall", "auc"]
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

        x = np.arange(len(folds))
        width = 0.2

        for i, metric in enumerate(metrics):
            values = [result[metric] for result in cv_results]
            ax1.bar(
                x + i * width,
                values,
                width,
                label=metric.upper(),
                color=colors[i],
                alpha=0.8,
            )

            # Add value labels
            for j, v in enumerate(values):
                ax1.text(
                    j + i * width,
                    v + 0.01,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax1.set_xlabel("Fold")
        ax1.set_ylabel("Score")
        ax1.set_title(
            "Cross-Validation Metrics Across Folds",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(folds)  # type: ignore
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)

        # Plot 2: ROC Curves for all folds (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        for i, fold_pred in enumerate(fold_predictions):
            fpr, tpr, _ = roc_curve(
                fold_pred["true_labels"], fold_pred["probabilities"]
            )
            auc_score = cv_results[i]["auc"]
            ax2.plot(
                fpr,
                tpr,
                alpha=0.7,
                linewidth=2,
                label=f"Fold {i + 1} (AUC = {auc_score:.3f})",
            )

        ax2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curves - All Folds", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect("equal")

        # Plot 3: Metric distributions (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        metric_data = {
            metric: [result[metric] for result in cv_results]
            for metric in metrics
        }
        box_plot = ax3.boxplot(
            [metric_data[metric] for metric in metrics],
            labels=[m.upper() for m in metrics],  # type: ignore
            patch_artist=True,
        )

        # Color the boxes
        colors_box = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
        for patch, color in zip(box_plot["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_ylabel("Score")
        ax3.set_title(
            "Metric Distributions Across Folds", fontsize=14, fontweight="bold"
        )
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)

        # Plot 4: Performance summary table (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("tight")
        ax4.axis("off")

        # Calculate summary statistics
        summary_data = []
        for metric in metrics:
            values = [result[metric] for result in cv_results]
            summary_data.append(
                [
                    metric.upper(),
                    f"{np.mean(values):.4f}",
                    f"{np.std(values):.4f}",
                    f"{np.min(values):.4f}",
                    f"{np.max(values):.4f}",
                ]
            )

        table = ax4.table(
            cellText=summary_data,
            colLabels=["Metric", "Mean", "Std", "Min", "Max"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],  # type: ignore
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title("Performance Summary", fontsize=14, fontweight="bold")

        # Plot 5: Training loss curves if available (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        # This would require storing loss history during training
        ax5.text(
            0.5,
            0.5,
            "Training Loss Curves\n(Enable logging to see)",
            ha="center",
            va="center",
            transform=ax5.transAxes,
            fontsize=12,
        )
        ax5.set_title("Training Progress", fontsize=14, fontweight="bold")
        ax5.axis("off")

        # Plot 6: Confusion matrix for best fold (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        best_fold_idx = np.argmax([result["f1"] for result in cv_results])
        best_fold_pred = fold_predictions[best_fold_idx]

        cm = confusion_matrix(
            best_fold_pred["true_labels"], best_fold_pred["predictions"]
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax6,
            xticklabels=["No Violation", "Violation"],
            yticklabels=["No Violation", "Violation"],
        )
        ax6.set_title(
            f"Confusion Matrix - Best Fold (Fold {best_fold_idx + 1})",
            fontsize=14,
            fontweight="bold",
        )
        ax6.set_xlabel("Predicted")
        ax6.set_ylabel("Actual")

        # Plot 7: Class distribution (bottom center)
        ax7 = fig.add_subplot(gs[2, 1])
        all_true_labels = np.concatenate(
            [fp["true_labels"] for fp in fold_predictions]
        )
        class_counts = [
            np.sum(all_true_labels == 0),
            np.sum(all_true_labels == 1),
        ]
        colors_pie = ["#FF6B6B", "#4ECDC4"]
        ax7.pie(
            class_counts,
            labels=["No Violation", "Violation"],
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
        )
        ax7.set_title(
            "Overall Class Distribution", fontsize=14, fontweight="bold"
        )

        # Plot 8: Fold-wise sample sizes (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        train_sizes = [result["train_size"] for result in cv_results]
        val_sizes = [result["val_size"] for result in cv_results]

        x = np.arange(len(folds))
        ax8.bar(
            x - 0.2,
            train_sizes,
            0.4,
            label="Training",
            color="#2E86AB",
            alpha=0.8,
        )
        ax8.bar(
            x + 0.2,
            val_sizes,
            0.4,
            label="Validation",
            color="#A23B72",
            alpha=0.8,
        )

        ax8.set_xlabel("Fold")
        ax8.set_ylabel("Number of Samples")
        ax8.set_title("Sample Sizes per Fold", fontsize=14, fontweight="bold")
        ax8.set_xticks(x)
        ax8.set_xticklabels(folds)  # type: ignore
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # Add overall title
        plt.suptitle(
            f"{model_name} Cross-Validation Analysis Dashboard",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"Comprehensive dashboard saved as {filename}")

    def evaluate_model(self, probs, true_labels, threshold=0.5):
        probs = np.array(probs).squeeze()
        true_labels = np.array(true_labels).astype(int)

        # derive predicted labels by thresholding
        pred_labels = np.argmax(probs, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="binary"
        )
        auc_score = roc_auc_score(true_labels, probs)
        cm = confusion_matrix(true_labels, pred_labels)

        return {
            "predictions": pred_labels,
            "probabilities": probs[:, 1],
            "true_labels": true_labels,
            "confusion_matrix": cm,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc_score,
            "classification_report": classification_report(
                true_labels, pred_labels
            ),
        }

    def print_cv_summary(self, cv_results):
        metrics = ["precision", "recall", "f1", "auc"]

        for metric in metrics:
            values = [result[metric] for result in cv_results]
            mean_val = np.mean(values)
            std_val = np.std(values)

            print(f"  Mean: {mean_val:.4f} ± {std_val:.4f}")
            print(f"  Range: {min(values):.4f} - {max(values):.4f}")
            print(f"  Fold values: {[f'{v:.4f}' for v in values]}")

        # Overall summary
        f1_scores = [result["f1"] for result in cv_results]
        auc_scores = [result["auc"] for result in cv_results]

        print(
            f"  Mean F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}"
        )
        print(
            f"  Mean AUC-ROC:  {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}"
        )

        stability = np.std(f1_scores)
        if stability < 0.03:
            stability_text = "Excellent"
        elif stability < 0.06:
            stability_text = "Good"
        elif stability < 0.1:
            stability_text = "Moderate"
        else:
            stability_text = "Variable"
        print(f"  Model Stability: {stability_text} (std: {stability:.4f})")


class InstructEval(JigsawEval):
    def cross_validate_with_data(self, data):
        data.drop_duplicates(
            subset=["body", "rule"], keep="first", inplace=True
        )
        skf = StratifiedKFold(
            n_splits=InstructConfig.N_SPLITS,
            shuffle=True,
            random_state=InstructConfig.RANDOM_STATE,
        )
        cv_results = []
        fold_predictions = []
        data = data.reset_index(drop=True)
        data["row_id"] = data.index

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(data, data["rule"]), 1
        ):
            # Split data
            train_df = data.iloc[train_idx].reset_index(drop=True)
            val_df = data.iloc[val_idx].reset_index(drop=True)

            # Initialize model for this fold
            fold_save_path = self.lora_path + f"_fold{fold}"

            trainer = Instruct(
                data_path=self.data_path,
                model_path=self.model_path,
                save_path=fold_save_path,
            )

            # hackyfix start training on subprocess to avoid cuda re-init issues during inference
            # TODO : need better fix
            p0 = mp.Process(target=trainer.train_with_data, args=(train_df,))
            p0.start()
            p0.join()
            # trainer.train_with_data(train_df)

            engine = InstructEngine(
                data_path=self.data_path,
                model_path=self.model_path,
                lora_path=fold_save_path,
                save_path=InstructConfig.out_file,
            )

            probs = engine.inference_with_data(val_df, return_preds=True)
            valid_labels = val_df["rule_violation"].tolist()
            # Evaluate
            fold_results = self.evaluate_model(
                probs,
                valid_labels,
            )

            # Store results
            cv_results.append(
                {
                    "fold": fold,
                    "precision": fold_results["precision"],
                    "recall": fold_results["recall"],
                    "f1": fold_results["f1"],
                    "auc": fold_results["auc"],
                    "train_size": len(train_df),
                    "val_size": len(val_df),
                }
            )

            # Store predictions for this fold
            fold_predictions.append(
                {
                    "fold": fold,
                    "true_labels": fold_results["true_labels"],
                    "predictions": fold_results["predictions"],
                    "probabilities": fold_results["probabilities"],
                    "rules": val_df["rule"].tolist(),
                }
            )

            print(f"Fold {fold} Results:")
            print(f"  Precision: {fold_results['precision']:.4f}")
            print(f"  Recall:    {fold_results['recall']:.4f}")
            print(f"  F1-Score:  {fold_results['f1']:.4f}")
            print(f"  AUC-ROC:   {fold_results['auc']:.4f}")

        return cv_results, fold_predictions

    def run(self):
        seed_everything(InstructConfig.RANDOM_STATE)

        dataframe = get_train_dataframe(InstructConfig.model_type)

        cv_results, fold_predictions = self.cross_validate_with_data(dataframe)

        # Print summary
        self.print_cv_summary(cv_results)

        self.create_dashboard(
            cv_results,
            fold_predictions,
            "Qwen 2.5 0.5b",
            "cv_dashboard_instruct.png",
        )

        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv("cross_validation_results_instruct.csv", index=False)

        # Save all predictions
        all_predictions = []
        for fold_pred in fold_predictions:
            fold_df = pd.DataFrame(
                {
                    "fold": fold_pred["fold"],
                    "true_label": fold_pred["true_labels"],
                    "predicted_label": fold_pred["predictions"],
                    "probability": fold_pred["probabilities"],
                    "rule": fold_pred["rules"],
                }
            )
            all_predictions.append(fold_df)

        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_df.to_csv("cv_predictions_instruct.csv", index=False)


class ChatEval(JigsawEval):
    def evaluate_with_data(self, data):
        data.drop_duplicates(
            subset=["body", "rule"], keep="first", inplace=True
        )

        results = []
        predictions = []
        data = data.reset_index(drop=True)
        data["row_id"] = data.index

        engine = ChatEngine(
            data_path=self.data_path,
            model_path=self.model_path,
            lora_path=self.lora_path,
            save_path=self.save_path,
        )

        probs = engine.inference_with_data(data, return_preds=True)
        valid_labels = data["rule_violation"].tolist()
        # Evaluate
        eval_results = self.evaluate_model(
            probs,
            valid_labels,
        )

        # Store results
        results.append(
            {
                "fold": 1,
                "precision": eval_results["precision"],
                "recall": eval_results["recall"],
                "f1": eval_results["f1"],
                "auc": eval_results["auc"],
                "train_size": 0,
                "val_size": len(data),
            }
        )

        # Store predictions
        predictions.append(
            {
                "fold": 1,
                "true_labels": eval_results["true_labels"],
                "predictions": eval_results["predictions"],
                "probabilities": eval_results["probabilities"],
                "rules": data["rule"].tolist(),
            }
        )

        print("Fold 1 Results:")
        print(f"  Precision: {eval_results['precision']:.4f}")
        print(f"  Recall:    {eval_results['recall']:.4f}")
        print(f"  F1-Score:  {eval_results['f1']:.4f}")
        print(f"  AUC-ROC:   {eval_results['auc']:.4f}")

        return results, predictions

    def run(self):
        seed_everything(ChatConfig.RANDOM_STATE)

        dataframe = get_train_dataframe(ChatConfig.model_type)

        results, predictions = self.evaluate_with_data(dataframe)

        # Print summary
        self.print_cv_summary(results)

        self.create_dashboard(
            results, predictions, "Qwen 2.5 14b", "dashboard_chat.png"
        )

        cv_df = pd.DataFrame(results)
        cv_df.to_csv("results_chat.csv", index=False)

        # Save all predictions
        all_predictions = []
        for pred in predictions:
            df = pd.DataFrame(
                {
                    "fold": pred["fold"],
                    "true_label": pred["true_labels"],
                    "predicted_label": pred["predictions"],
                    "probability": pred["probabilities"],
                    "rule": pred["rules"],
                }
            )
            all_predictions.append(df)

        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_df.to_csv("predictions_chat.csv", index=False)


class RobertaEval(JigsawEval):
    def cross_validate_with_data(self, data):
        data.drop_duplicates(
            subset=["body", "rule"], keep="first", inplace=True
        )
        skf = StratifiedKFold(
            n_splits=RobertaConfig.N_SPLITS,
            shuffle=True,
            random_state=RobertaConfig.RANDOM_STATE,
        )
        cv_results = []
        fold_predictions = []
        data = data.reset_index(drop=True)
        data["row_id"] = data.index

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(data, data["rule"]), 1
        ):
            # Split data
            train_df = data.iloc[train_idx].reset_index(drop=True)
            val_df = data.iloc[val_idx].reset_index(drop=True)

            # Initialize model for this fold
            fold_save_path = self.save_path + f"_fold{fold}"

            trainer = RobertaBase(
                data_path=self.data_path,
                model_path=self.model_path,
                save_path=fold_save_path,
            )

            trainer.train_with_data(train_df)

            engine = RobertaEngine(
                data_path=self.data_path,
                model_path=fold_save_path,
                save_path=RobertaConfig.out_file,
            )

            probs = engine.inference_with_data(val_df, return_preds=True)
            valid_labels = val_df["rule_violation"].tolist()
            # Evaluate
            fold_results = self.evaluate_model(
                probs,
                valid_labels,
            )

            # Store results
            cv_results.append(
                {
                    "fold": fold,
                    "precision": fold_results["precision"],
                    "recall": fold_results["recall"],
                    "f1": fold_results["f1"],
                    "auc": fold_results["auc"],
                    "train_size": len(train_df),
                    "val_size": len(val_df),
                }
            )

            # Store predictions for this fold
            fold_predictions.append(
                {
                    "fold": fold,
                    "true_labels": fold_results["true_labels"],
                    "predictions": fold_results["predictions"],
                    "probabilities": fold_results["probabilities"],
                    "rules": val_df["rule"].tolist(),
                }
            )

            print(f"Fold {fold} Results:")
            print(f"  Precision: {fold_results['precision']:.4f}")
            print(f"  Recall:    {fold_results['recall']:.4f}")
            print(f"  F1-Score:  {fold_results['f1']:.4f}")
            print(f"  AUC-ROC:   {fold_results['auc']:.4f}")

        return cv_results, fold_predictions

    def run(self):
        seed_everything(RobertaConfig.RANDOM_STATE)

        dataframe, _ = get_train_dataframe(
            RobertaConfig.model_type
        )  # use only train

        cv_results, fold_predictions = self.cross_validate_with_data(dataframe)

        # Print summary
        self.print_cv_summary(cv_results)

        self.create_dashboard(
            cv_results,
            fold_predictions,
            "Roberta Base",
            "cv_dashboard_roberta.png",
        )

        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv("cross_validation_results_roberta.csv", index=False)

        # Save all predictions
        all_predictions = []
        for fold_pred in fold_predictions:
            fold_df = pd.DataFrame(
                {
                    "fold": fold_pred["fold"],
                    "true_label": fold_pred["true_labels"],
                    "predicted_label": fold_pred["predictions"],
                    "probability": fold_pred["probabilities"],
                    "rule": fold_pred["rules"],
                }
            )
            all_predictions.append(fold_df)

        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_df.to_csv("cv_predictions_roberta.csv", index=False)


class DebertaEval(JigsawEval):
    def cross_validate_with_data(self, data):
        data.drop_duplicates(
            subset=["body", "rule"], keep="first", inplace=True
        )
        skf = StratifiedKFold(
            n_splits=DebertaConfig.N_SPLITS,
            shuffle=True,
            random_state=DebertaConfig.RANDOM_STATE,
        )
        cv_results = []
        fold_predictions = []
        data = data.reset_index(drop=True)
        data["row_id"] = data.index

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(data, data["rule"]), 1
        ):
            # Split data
            train_df = data.iloc[train_idx].reset_index(drop=True)
            val_df = data.iloc[val_idx].reset_index(drop=True)

            # Initialize model for this fold
            fold_save_path = self.save_path + f"_fold{fold}"

            trainer = DebertaBase(
                data_path=self.data_path,
                model_path=self.model_path,
                save_path=fold_save_path,
            )
            trainer.train_with_data(train_df)

            engine = DebertaEngine(
                data_path=self.data_path,
                model_path=fold_save_path,
                save_path=DebertaConfig.out_file,
            )

            probs = engine.inference_with_data(val_df, return_preds=True)
            valid_labels = val_df["rule_violation"].tolist()
            # Evaluate
            fold_results = self.evaluate_model(
                probs,
                valid_labels,
            )

            # Store results
            cv_results.append(
                {
                    "fold": fold,
                    "precision": fold_results["precision"],
                    "recall": fold_results["recall"],
                    "f1": fold_results["f1"],
                    "auc": fold_results["auc"],
                    "train_size": len(train_df),
                    "val_size": len(val_df),
                }
            )

            # Store predictions for this fold
            fold_predictions.append(
                {
                    "fold": fold,
                    "true_labels": fold_results["true_labels"],
                    "predictions": fold_results["predictions"],
                    "probabilities": fold_results["probabilities"],
                    "rules": val_df["rule"].tolist(),
                }
            )

            print(f"Fold {fold} Results:")
            print(f"  Precision: {fold_results['precision']:.4f}")
            print(f"  Recall:    {fold_results['recall']:.4f}")
            print(f"  F1-Score:  {fold_results['f1']:.4f}")
            print(f"  AUC-ROC:   {fold_results['auc']:.4f}")

        return cv_results, fold_predictions

    def run(self):
        seed_everything(DebertaConfig.RANDOM_STATE)

        dataframe = get_train_dataframe(DebertaConfig.model_type)

        cv_results, fold_predictions = self.cross_validate_with_data(dataframe)

        # Print summary
        self.print_cv_summary(cv_results)

        self.create_dashboard(
            cv_results,
            fold_predictions,
            "DeBERTa v3 Base",
            "cv_dashboard_deberta.png",
        )

        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv("cross_validation_results_deberta.csv", index=False)

        # Save all predictions
        all_predictions = []
        for fold_pred in fold_predictions:
            fold_df = pd.DataFrame(
                {
                    "fold": fold_pred["fold"],
                    "true_label": fold_pred["true_labels"],
                    "predicted_label": fold_pred["predictions"],
                    "probability": fold_pred["probabilities"],
                    "rule": fold_pred["rules"],
                }
            )
            all_predictions.append(fold_df)

        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_df.to_csv("cv_predictions_deberta.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str)

    args = parser.parse_args()

    if args.type == InstructConfig.model_type:
        evaluator: JigsawEval = InstructEval(
            data_path=InstructConfig.data_path,
            model_path=InstructConfig.model_path,
            lora_path=InstructConfig.lora_path,
            save_path=InstructConfig.out_file,
        )
        evaluator.run()
    elif args.type == ChatConfig.model_type:
        evaluator = ChatEval(
            data_path=ChatConfig.data_path,
            model_path=ChatConfig.model_path,
            lora_path=ChatConfig.lora_path,
            save_path=ChatConfig.out_file,
        )
        evaluator.run()
    elif args.type == RobertaConfig.model_type:
        evaluator = RobertaEval(
            data_path=RobertaConfig.data_path,
            model_path=RobertaConfig.model_path,
            save_path=RobertaConfig.ckpt_path,
        )
        evaluator.run()
    elif args.type == DebertaConfig.model_type:
        evaluator = DebertaEval(
            data_path=DebertaConfig.data_path,
            model_path=DebertaConfig.model_path,
            save_path=DebertaConfig.ckpt_path,
        )
        evaluator.run()
    else:
        raise AttributeError("Invalid evaluation type")
