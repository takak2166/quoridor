"""Lightweight self-play league runner for comparing difficulty presets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from app.infrastructure.rl.eval_selfplay import run_eval


@dataclass(frozen=True)
class LeagueMatchup:
    difficulty_a: str
    difficulty_b: str
    games: int = 20


DEFAULT_LEAGUE: tuple[LeagueMatchup, ...] = (
    LeagueMatchup("hard", "random", games=10),
    LeagueMatchup("hard", "very_easy", games=10),
    LeagueMatchup("hard", "easy", games=10),
    LeagueMatchup("hard", "normal", games=20),
)


def run_league(
    matchups: tuple[LeagueMatchup, ...] = DEFAULT_LEAGUE,
    *,
    max_p99_ms: float | None = 3000,
) -> list[dict[str, float | str]]:
    results: list[dict[str, float | str]] = []
    for matchup in matchups:
        stats = run_eval(
            matchup.games,
            matchup.difficulty_a,
            matchup.difficulty_b,
            max_p99_ms=max_p99_ms,
        )
        results.append(
            {
                "difficulty_a": matchup.difficulty_a,
                "difficulty_b": matchup.difficulty_b,
                **stats,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-p99-ms", type=float, default=3000)
    args = parser.parse_args()

    results = run_league(DEFAULT_LEAGUE, max_p99_ms=args.max_p99_ms)
    for row in results:
        print(
            f"{row['difficulty_a']} vs {row['difficulty_b']}: "
            f"win_rate={row['win_rate']:.1%} "
            f"(black={row['win_rate_as_black']:.1%}, white={row['win_rate_as_white']:.1%}) "
            f"p99={row['p99_ms']:.0f}ms"
        )


if __name__ == "__main__":
    main()
