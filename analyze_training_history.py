import argparse
import csv
import os

import matplotlib.pyplot as plt


def load_training_history(csv_file = "q_learning_training_history_v2.csv"):
    training_episodes = []
    scores = []
    wins = []
    steps = []
    epsilon_values = []
    alpha_values = []
    gamma_values = []

    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "training_episode",
            "score",
            "win",
            "steps",
            "epsilon",
            "alpha",
            "gamma"
        }

        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                "Missing CSV columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            if row.get("mode", "training") != "training":
                continue

            try:
                training_episode = int(row["training_episode"])
                score = float(row["score"])
                episode_steps = int(row["steps"])
                epsilon = float(row["epsilon"])
                alpha = float(row["alpha"])
                gamma = float(row["gamma"])
            except (TypeError, ValueError):
                print(
                    "Skip invalid row:",
                    row_number
                )
                continue

            win = str(row["win"]).strip().lower() == "true"

            training_episodes.append(training_episode)
            scores.append(score)
            wins.append(win)
            steps.append(episode_steps)
            epsilon_values.append(epsilon)
            alpha_values.append(alpha)
            gamma_values.append(gamma)

    return (
        training_episodes,
        scores,
        wins,
        steps,
        epsilon_values,
        alpha_values,
        gamma_values
    )


def get_parameter_legend(
    epsilon_values,
    alpha_values,
    gamma_values
):
    epsilon_text = (
        f"{min(epsilon_values):.4f} - "
        f"{max(epsilon_values):.4f}"
    )

    if min(alpha_values) == max(alpha_values):
        alpha_text = f"{alpha_values[0]:.4f}"
    else:
        alpha_text = (
            f"{min(alpha_values):.4f} - "
            f"{max(alpha_values):.4f}"
        )

    if min(gamma_values) == max(gamma_values):
        gamma_text = f"{gamma_values[0]:.4f}"
    else:
        gamma_text = (
            f"{min(gamma_values):.4f} - "
            f"{max(gamma_values):.4f}"
        )

    return (
        "epsilon          alpha          gamma\n"
        f"{epsilon_text}    {alpha_text}    {gamma_text}"
    )


def draw_score_distribution(
    training_episodes,
    scores,
    wins,
    steps,
    epsilon_values,
    alpha_values,
    gamma_values,
    output_file
):
    if not training_episodes:
        raise ValueError("No valid training records found.")

    win_episodes = []
    win_scores = []
    win_steps = []
    lose_episodes = []
    lose_scores = []
    lose_steps = []

    for training_episode, score, win, episode_steps in zip(
        training_episodes,
        scores,
        wins,
        steps
    ):
        if win:
            win_episodes.append(training_episode)
            win_scores.append(score)
            win_steps.append(episode_steps)
        else:
            lose_episodes.append(training_episode)
            lose_scores.append(score)
            lose_steps.append(episode_steps)

    parameter_legend = get_parameter_legend(
        epsilon_values,
        alpha_values,
        gamma_values
    )

    figure = plt.figure(figsize=(15, 6))

    score_axes = figure.add_subplot(1, 2, 1)
    steps_axes = figure.add_subplot(1, 2, 2)

    if lose_episodes:
        score_axes.scatter(
            lose_episodes,
            lose_scores,
            color="#B00020",
            alpha=0.20,
            s=4,
            label="Lose"
        )

        steps_axes.scatter(
            lose_episodes,
            lose_steps,
            color="#B00020",
            alpha=0.20,
            s=4,
            label="Lose"
        )

    if win_episodes:
        score_axes.scatter(
            win_episodes,
            win_scores,
            color="#006400",
            alpha=0.20,
            s=4,
            label="Win"
        )

        steps_axes.scatter(
            win_episodes,
            win_steps,
            color="#006400",
            alpha=0.20,
            s=4,
            label="Win"
        )

    score_axes.set_title(
        "Training Episode and Score Distribution"
    )
    score_axes.set_xlabel("Training Episode")
    score_axes.set_ylabel("Score")
    score_axes.grid(True, alpha=0.3)

    score_legend = score_axes.legend(
        title=parameter_legend,
        markerscale=4
    )
    score_legend.get_title().set_fontfamily("monospace")

    steps_axes.set_title(
        "Training Episode and Steps Distribution"
    )
    steps_axes.set_xlabel("Training Episode")
    steps_axes.set_ylabel("Steps")
    steps_axes.grid(True, alpha=0.3)

    steps_legend = steps_axes.legend(
        title=parameter_legend,
        markerscale=4
    )
    steps_legend.get_title().set_fontfamily(
        "monospace"
    )

    figure.tight_layout(w_pad=7.0)

    figure.savefig(
        output_file,
        dpi=200
    )

    print("Saved image:", output_file)

    plt.show()


def main():
    current_directory = os.path.dirname(
        os.path.abspath(__file__)
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=os.path.join(
            current_directory,
            "q_learning_training_history_v2.csv"
        )
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            current_directory,
            "training_score_distribution.png"
        )
    )

    arguments = parser.parse_args()

    (
        training_episodes,
        scores,
        wins,
        steps,
        epsilon_values,
        alpha_values,
        gamma_values
    ) = load_training_history(arguments.csv_file)

    draw_score_distribution(
        training_episodes,
        scores,
        wins,
        steps,
        epsilon_values,
        alpha_values,
        gamma_values,
        arguments.output
    )


if __name__ == "__main__":
    main()
