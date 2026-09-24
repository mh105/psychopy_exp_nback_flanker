# /// script
# requires-python = "==3.12.*"
# dependencies = ["z3-solver==5.1.0.0"]
# ///
"""Generate the final fixed stimulus lists for the n-back flanker task.

Run from the project root with:

    uv run resource/generate_stimulus_lists.py

The script writes three reproducible ListVersions to ``resource/stimulus_lists``
and creates ``resource/generate_stimulus_lists_report.html``. The generated
HTML report is the authoritative description of the constructs, condition
definitions, list constraints, validation policy, and observed main-list
properties. Existing CSVs must match the deterministic output and are never
overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
from collections import Counter
from dataclasses import dataclass
from html import escape
from io import StringIO
from pathlib import Path
from typing import Iterable, Sequence

try:
    from z3 import If, Int, Or, Solver, Sum, sat
except ImportError as exc:  # pragma: no cover - reached only outside the uv command
    raise SystemExit(
        "z3-solver is required. Run: uv run resource/generate_stimulus_lists.py"
    ) from exc


LETTERS = ("S", "H", "C", "F")
NON_TARGET_LETTERS = (0, 2, 3)
TARGET = 1  # H
TASKS = ("0back", "1back")
BLOCKS = ("practice", "main")
VERSIONS = (1, 2, 3)
TYPE_NAMES = ("A1", "A2", "B1", "B2")
BINARY_FIELDS = (
    "PerceptualCongruency",
    "BehavioralCongruency",
    "CorrectResponse",
)
HEADER = (
    "TrialNumber",
    "CenterLetter",
    "FlankerLetter",
    "PerceptualCongruency",
    "BehavioralCongruency",
    "CorrectResponse",
    "ConditionType",
)

# Values are (PerceptualCongruency, BehavioralCongruency, CorrectResponse).
TYPE_FIELDS = {
    "0back": {
        0: ("yes", "yes", "no"),   # A1
        1: ("no", "no", "no"),     # A2
        2: ("yes", "yes", "yes"),  # B1
        3: ("no", "no", "yes"),    # B2
    },
    "1back": {
        0: ("no", "yes", "no"),    # A1
        1: ("yes", "no", "no"),    # A2
        2: ("yes", "yes", "yes"),  # B1
        3: ("no", "no", "yes"),    # B2
    },
}

# Fixed prevalidated trial-type orders. Digits 0..3 denote A1, A2, B1, B2.
# The main orders are deliberately shared across tasks within each version.
PRACTICE_0BACK_ORDERS = (
    "03220112331021203310",
    "11303213002320322011",
    "23300032113021201132",
)
PRACTICE_1BACK_ORDERS = (
    "13220312300123312001",
    "22312331031200130012",
    "10023301200312231231",
)
MAIN_ORDERS = (
    "1033103310331220122033100322012201220122012231003103310331033103"
    "3122013301220122312201330122013301220133012201330133012201220130"
    "12201033122013013301330122012201",
    "3310331033103312201220122300122013301301220122012331033103310031"
    "0331220122012231033103312201220122012013301330122012201220122013"
    "30122012201220131033103310031033",
    "1233103310331033100322012201330122012231003103310331033122012201"
    "3301220130122013301330122013301220103312001220331220122013013301"
    "33012201220133012201223122012201",
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = ROOT / "stimulus_lists"
REPORT_PATH = ROOT / "generate_stimulus_lists_report.html"


@dataclass(frozen=True)
class ListKey:
    task: str
    block: str
    version: int

    @property
    def n_scored(self) -> int:
        return 20 if self.block == "practice" else 160

    @property
    def filename(self) -> str:
        return f"ListVersion{self.version}_{self.task}_{self.block}.csv"


@dataclass(frozen=True)
class GeneratedList:
    key: ListKey
    types: tuple[int, ...]
    centers: tuple[int, ...]
    flankers: tuple[int, ...]


@dataclass(frozen=True)
class ValidatedList:
    key: ListKey
    rows: tuple[tuple[str, ...], ...]
    types: tuple[int, ...]
    complete_rows: tuple[tuple[str, ...], ...]
    binary_max_runs: dict[str, int]
    raw_max_runs: dict[str, int]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def indicator(condition):
    return If(condition, 1, 0)


def max_run(values: Sequence[object]) -> int:
    best = 0
    run = 0
    previous = object()
    for value in values:
        if value == previous:
            run += 1
        else:
            previous = value
            run = 1
        best = max(best, run)
    return best


def exact_run_episodes(values: Sequence[object]) -> Counter[tuple[object, int]]:
    episodes: Counter[tuple[object, int]] = Counter()
    if not values:
        return episodes
    previous = values[0]
    length = 1
    for value in values[1:]:
        if value == previous:
            length += 1
        else:
            episodes[(previous, length)] += 1
            previous = value
            length = 1
    episodes[(previous, length)] += 1
    return episodes


def transition_counts(values: Sequence[str]) -> dict[tuple[str, str], int]:
    counts = {
        (left, right): 0
        for left in ("yes", "no")
        for right in ("yes", "no")
    }
    for pair in zip(values, values[1:]):
        counts[pair] += 1
    return counts


def repeated_divisor_period(values: Sequence[object]) -> int:
    size = len(values)
    for period in range(1, size):
        if size % period == 0 and all(
            values[index] == values[index % period] for index in range(size)
        ):
            return period
    return size


def hamming(left: Sequence[object], right: Sequence[object]) -> int:
    require(len(left) == len(right), "Cannot compare sequences of different lengths")
    return sum(a != b for a, b in zip(left, right))


def dihedral_equivalent(left: Sequence[object], right: Sequence[object]) -> bool:
    left_tuple = tuple(left)
    right_tuple = tuple(right)
    require(
        len(left_tuple) == len(right_tuple),
        "Cannot compare sequences of different lengths",
    )
    reversed_left = left_tuple[::-1]
    for offset in range(len(left_tuple)):
        if right_tuple == left_tuple[offset:] + left_tuple[:offset]:
            return True
        if right_tuple == reversed_left[offset:] + reversed_left[:offset]:
            return True
    return False


def near_balance_expected(size: int, alphabet_size: int) -> list[int]:
    low, remainder = divmod(size, alphabet_size)
    return sorted(
        [low] * (alphabet_size - remainder) + [low + 1] * remainder
    )


def require_near_balance(
    values: Sequence[str], alphabet: Sequence[str], context: str
) -> None:
    observed = sorted(Counter(values)[item] for item in alphabet)
    expected = near_balance_expected(len(values), len(alphabet))
    require(
        observed == expected,
        f"{context}: identity counts {observed}, expected {expected}",
    )


def add_near_balance_constraints(
    solver: Solver,
    values: Sequence[object],
    positions: Sequence[int],
    alphabet: Sequence[int],
) -> None:
    low, remainder = divmod(len(positions), len(alphabet))
    high = low + bool(remainder)
    for letter in alphabet:
        count = Sum(*[indicator(values[index] == letter) for index in positions])
        solver.add(count >= low, count <= high)


def add_value_run_cap(solver: Solver, values: Sequence[object], cap: int) -> None:
    for start in range(len(values) - cap):
        solver.add(
            Or(
                *[
                    values[start + offset] != values[start]
                    for offset in range(1, cap + 1)
                ]
            )
        )


def add_pair_run_cap(
    solver: Solver,
    centers: Sequence[object],
    flankers: Sequence[object],
    cap: int,
) -> None:
    for start in range(len(centers) - cap):
        solver.add(
            Or(
                *[
                    Or(
                        centers[start + offset] != centers[start],
                        flankers[start + offset] != flankers[start],
                    )
                    for offset in range(1, cap + 1)
                ]
            )
        )


def constrain_domains(solver: Solver, values: Iterable[object]) -> None:
    for value in values:
        solver.add(value >= 0, value < len(LETTERS))


def solve_1back_identities(
    key: ListKey, types: tuple[int, ...]
) -> GeneratedList:
    n_scored = len(types)
    centers = [Int(f"c_{index}") for index in range(n_scored + 1)]
    flankers = [Int(f"f_{index}") for index in range(n_scored + 1)]
    solver = Solver()
    seed = (
        20266000 + key.version * 100 + ord("D")
        if key.block == "main"
        else 20260901 + key.version
    )
    solver.set(random_seed=seed)
    constrain_domains(solver, centers + flankers)

    # Trial 0 is the displayed, response-free previous state.
    # Letter labels are otherwise interchangeable, so pin one arbitrary pair
    # to remove equivalent solver branches.
    solver.add(centers[0] == 0, flankers[0] == 1)

    for scored_index, trial_type in enumerate(types, start=1):
        previous_center = centers[scored_index - 1]
        previous_flanker = flankers[scored_index - 1]
        center = centers[scored_index]
        flanker = flankers[scored_index]
        if trial_type == 0:  # A1
            solver.add(
                center != previous_center,
                flanker != center,
                flanker != previous_center,
                flanker != previous_flanker,
            )
        elif trial_type == 1:  # A2
            solver.add(
                center != previous_center,
                center == previous_flanker,
                flanker == center,
            )
        elif trial_type == 2:  # B1
            solver.add(center == previous_center, flanker == center)
        else:  # B2
            solver.add(
                center == previous_center,
                flanker != center,
                flanker != previous_flanker,
            )

    scored_positions = list(range(1, n_scored + 1))
    for trial_type in range(4):
        positions = [
            index + 1 for index, value in enumerate(types) if value == trial_type
        ]
        for values in (centers, flankers):
            add_near_balance_constraints(
                solver, values, positions, tuple(range(len(LETTERS)))
            )
    for values in (centers, flankers):
        add_near_balance_constraints(
            solver,
            values,
            scored_positions,
            tuple(range(len(LETTERS))),
        )

    # Raw visual runs include trial 0.
    add_value_run_cap(solver, centers, 4)
    add_value_run_cap(solver, flankers, 4)
    add_pair_run_cap(solver, centers, flankers, 3)

    status = solver.check()
    require(status == sat, f"{key.filename}: identity solver returned {status}")
    model = solver.model()
    return GeneratedList(
        key=key,
        types=types,
        centers=tuple(model.eval(value).as_long() for value in centers),
        flankers=tuple(model.eval(value).as_long() for value in flankers),
    )


def solve_0back_identities(
    key: ListKey, types: tuple[int, ...]
) -> GeneratedList:
    n_scored = len(types)
    centers = [Int(f"z_c_{index}") for index in range(n_scored)]
    flankers = [Int(f"z_f_{index}") for index in range(n_scored)]
    solver = Solver()
    seed = (
        20265100 + key.version
        if key.block == "main"
        else 20260902 + key.version
    )
    solver.set(random_seed=seed)
    constrain_domains(solver, centers + flankers)

    for index, trial_type in enumerate(types):
        center = centers[index]
        flanker = flankers[index]
        if trial_type == 0:  # A1
            solver.add(center != TARGET, flanker == center)
        elif trial_type == 1:  # A2
            solver.add(center != TARGET, flanker == TARGET)
        elif trial_type == 2:  # B1
            solver.add(center == TARGET, flanker == TARGET)
        else:  # B2
            solver.add(center == TARGET, flanker != TARGET)

    variable_cells = (
        (0, centers),
        (0, flankers),
        (1, centers),
        (3, flankers),
    )
    for trial_type, values in variable_cells:
        positions = [
            index for index, value in enumerate(types) if value == trial_type
        ]
        add_near_balance_constraints(
            solver, values, positions, NON_TARGET_LETTERS
        )

    all_positions = list(range(n_scored))
    for values in (centers, flankers):
        non_target_positions = [
            index
            for index in all_positions
            if types[index] in (
                (0, 1) if values is centers else (0, 3)
            )
        ]
        add_near_balance_constraints(
            solver, values, non_target_positions, NON_TARGET_LETTERS
        )

    add_value_run_cap(solver, centers, 4)
    add_value_run_cap(solver, flankers, 4)
    add_pair_run_cap(solver, centers, flankers, 3)

    status = solver.check()
    require(status == sat, f"{key.filename}: identity solver returned {status}")
    model = solver.model()
    return GeneratedList(
        key=key,
        types=types,
        centers=tuple(model.eval(value).as_long() for value in centers),
        flankers=tuple(model.eval(value).as_long() for value in flankers),
    )


def parse_type_order(
    encoded: str, expected_size: int, context: str
) -> tuple[int, ...]:
    require(len(encoded) == expected_size, f"{context}: expected {expected_size} types")
    require(set(encoded) <= set("0123"), f"{context}: invalid type digit")
    return tuple(int(value) for value in encoded)


def projected_fields(task: str, types: Sequence[int]) -> dict[str, list[str]]:
    return {
        field: [TYPE_FIELDS[task][trial_type][field_index] for trial_type in types]
        for field_index, field in enumerate(BINARY_FIELDS)
    }


def validate_type_order(
    types: tuple[int, ...], task: str, block: str, context: str
) -> None:
    n_scored = 20 if block == "practice" else 160
    quota = n_scored // 4
    require(len(types) == n_scored, f"{context}: wrong type-order length")
    require(
        Counter(types) == Counter({value: quota for value in range(4)}),
        f"{context}: type counts are not equal",
    )
    if task == "1back":
        for index, trial_type in enumerate(types[1:], start=1):
            require(
                trial_type != 1 or types[index - 1] in (0, 3),
                f"{context}: A2 at trial {index + 1} has an impossible predecessor",
            )
    expected_transitions = (
        [4, 5, 5, 5] if block == "practice" else [39, 40, 40, 40]
    )
    for field, values in projected_fields(task, types).items():
        require(
            Counter(values) == Counter({"yes": n_scored // 2, "no": n_scored // 2}),
            f"{context}: {field} is not 50/50",
        )
        require(max_run(values) <= 3, f"{context}: {field} run exceeds 3")
        counts = transition_counts(values)
        require(
            sorted(counts.values()) == expected_transitions,
            f"{context}: {field} transition counts {counts}",
        )
    if block == "main":
        response_episodes = exact_run_episodes(
            projected_fields(task, types)["CorrectResponse"]
        )
        response_run3 = {
            value: response_episodes[(value, 3)] for value in ("yes", "no")
        }
        require(
            response_run3["yes"] == response_run3["no"],
            f"{context}: CorrectResponse run-3 counts differ by value "
            f"{response_run3}",
        )
        require(
            2 <= response_run3["yes"] <= 4,
            f"{context}: CorrectResponse run-3 count per value "
            f"{response_run3['yes']} is not between 2 and 4",
        )
        response_run3_total = sum(response_run3.values())
        require(
            response_run3_total < 10,
            f"{context}: pooled exact CorrectResponse run-3 episodes "
            f"{response_run3_total} is not below 10",
        )
        response_run2 = {
            value: response_episodes[(value, 2)] for value in ("yes", "no")
        }
        require(
            abs(response_run2["yes"] - response_run2["no"]) <= 5,
            f"{context}: CorrectResponse run-2 counts differ by more than 5 "
            f"{response_run2}",
        )
        for quarter in range(4):
            counts = Counter(types[quarter * 40 : (quarter + 1) * 40])
            require(
                all(8 <= counts[value] <= 12 for value in range(4)),
                f"{context}: quarter {quarter + 1} type counts {counts}",
            )
    require(
        repeated_divisor_period(types) == n_scored,
        f"{context}: type order repeats a shorter motif",
    )


def validate_order_group_distinctness(
    orders: Sequence[tuple[int, ...]], block: str, context: str
) -> None:
    require(
        len(orders) == len(VERSIONS),
        f"{context}: expected {len(VERSIONS)} ListVersions",
    )
    threshold = (20 if block == "practice" else 160) // 2
    for (left_version, left), (right_version, right) in itertools.combinations(
        zip(VERSIONS, orders), 2
    ):
        pair_context = f"{context} ListVersion{left_version}/{right_version}"
        require(
            hamming(left, right) >= threshold,
            f"{pair_context}: type Hamming distance below {threshold}",
        )
        require(
            not dihedral_equivalent(left, right),
            f"{pair_context}: type orders are rotation/reversal equivalent",
        )
        left_responses = projected_fields("0back", left)["CorrectResponse"]
        right_responses = projected_fields("0back", right)["CorrectResponse"]
        require(
            hamming(left_responses, right_responses) >= threshold,
            f"{pair_context}: CorrectResponse Hamming distance below {threshold}",
        )


def parse_and_validate_practice_orders(
    task: str, encoded_orders: Sequence[str]
) -> tuple[tuple[int, ...], ...]:
    orders = tuple(
        parse_type_order(
            encoded,
            20,
            f"{task} practice ListVersion{version}",
        )
        for version, encoded in zip(VERSIONS, encoded_orders)
    )
    for version, order in zip(VERSIONS, orders):
        validate_type_order(
            order,
            task,
            "practice",
            f"{task} practice ListVersion{version}",
        )
    validate_order_group_distinctness(orders, "practice", f"{task} practice")
    return orders


def parse_and_validate_main_orders() -> tuple[tuple[int, ...], ...]:
    orders = tuple(
        parse_type_order(encoded, 160, f"main ListVersion{version}")
        for version, encoded in zip(VERSIONS, MAIN_ORDERS)
    )
    for version, order in zip(VERSIONS, orders):
        for task in TASKS:
            validate_type_order(
                order,
                task,
                "main",
                f"{task} main ListVersion{version}",
            )
    validate_order_group_distinctness(orders, "main", "shared main")
    response_run2_by_version = []
    for order in orders:
        responses = projected_fields("0back", order)["CorrectResponse"]
        episodes = exact_run_episodes(responses)
        response_run2_by_version.append(
            (episodes[("yes", 2)], episodes[("no", 2)])
        )
    require(
        len(set(response_run2_by_version)) == len(VERSIONS),
        "shared main: ListVersions have identical CorrectResponse run-2 counts",
    )
    return orders


def build_all_lists() -> list[GeneratedList]:
    zero_practice = parse_and_validate_practice_orders(
        "0back", PRACTICE_0BACK_ORDERS
    )
    one_practice = parse_and_validate_practice_orders(
        "1back", PRACTICE_1BACK_ORDERS
    )
    main_orders = parse_and_validate_main_orders()

    generated: list[GeneratedList] = []
    for version_index, version in enumerate(VERSIONS):
        generated.append(
            solve_0back_identities(
                ListKey("0back", "practice", version),
                zero_practice[version_index],
            )
        )
        generated.append(
            solve_0back_identities(
                ListKey("0back", "main", version),
                main_orders[version_index],
            )
        )
        generated.append(
            solve_1back_identities(
                ListKey("1back", "practice", version),
                one_practice[version_index],
            )
        )
        generated.append(
            solve_1back_identities(
                ListKey("1back", "main", version),
                main_orders[version_index],
            )
        )
    return generated


def csv_rows(generated: GeneratedList) -> list[list[object]]:
    rows: list[list[object]] = []
    display_offset = 1 if generated.key.task == "1back" else 0
    if display_offset:
        rows.append(
            [
                0,
                LETTERS[generated.centers[0]],
                LETTERS[generated.flankers[0]],
                "no",
                "",
                "",
                "",
            ]
        )
    for trial_number, trial_type in enumerate(generated.types, start=1):
        display_index = trial_number - 1 + display_offset
        perceptual, behavioral, correct = TYPE_FIELDS[generated.key.task][trial_type]
        rows.append(
            [
                trial_number,
                LETTERS[generated.centers[display_index]],
                LETTERS[generated.flankers[display_index]],
                perceptual,
                behavioral,
                correct,
                TYPE_NAMES[trial_type],
            ]
        )
    return rows


def write_lists(
    generated_lists: Sequence[GeneratedList],
) -> dict[ListKey, Path]:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    paths: dict[ListKey, Path] = {}
    for generated in generated_lists:
        path = OUTPUT_DIRECTORY / generated.key.filename
        buffer = StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(csv_rows(generated))
        content = buffer.getvalue().encode("utf-8")
        if path.exists():
            require(
                path.read_bytes() == content,
                f"{path.name}: existing CSV differs from generated output; "
                "refusing to overwrite it",
            )
        else:
            with path.open("xb") as handle:
                handle.write(content)
        paths[generated.key] = path
    expected_paths = set(paths.values())
    actual_paths = set(OUTPUT_DIRECTORY.glob("*.csv"))
    require(
        actual_paths == expected_paths,
        "stimulus_lists contains unexpected CSV files; preserve or remove them before rerunning",
    )
    return paths


def rows_as_dicts(
    rows: Sequence[tuple[str, ...]],
) -> list[dict[str, str]]:
    return [dict(zip(HEADER, row)) for row in rows]


def validate_task_rules(
    task: str,
    rows: Sequence[dict[str, str]],
    scored_offset: int,
    types: Sequence[int],
    context: str,
) -> None:
    for scored_index, trial_type in enumerate(types):
        display_index = scored_index + scored_offset
        center = rows[display_index]["CenterLetter"]
        flanker = rows[display_index]["FlankerLetter"]
        if task == "0back":
            if trial_type == 0:
                valid = center != "H" and flanker == center
            elif trial_type == 1:
                valid = center != "H" and flanker == "H"
            elif trial_type == 2:
                valid = center == "H" and flanker == "H"
            else:
                valid = center == "H" and flanker != "H"
        else:
            previous_center = rows[display_index - 1]["CenterLetter"]
            previous_flanker = rows[display_index - 1]["FlankerLetter"]
            if trial_type == 0:
                valid = (
                    center != previous_center
                    and flanker
                    not in {center, previous_center, previous_flanker}
                )
            elif trial_type == 1:
                valid = (
                    center != previous_center
                    and center == previous_flanker
                    and flanker == center
                )
            elif trial_type == 2:
                valid = center == previous_center and flanker == center
            else:
                valid = (
                    center == previous_center
                    and flanker not in {center, previous_flanker}
                )
        require(
            valid,
            f"{context}: task rule failed at TrialNumber "
            f"{scored_index + 1} ({TYPE_NAMES[trial_type]})",
        )


def validate_identity_balance(
    task: str,
    scored_rows: Sequence[dict[str, str]],
    types: Sequence[int],
    context: str,
) -> None:
    if task == "1back":
        for trial_type in range(4):
            indices = [
                index for index, value in enumerate(types) if value == trial_type
            ]
            for field in ("CenterLetter", "FlankerLetter"):
                values = [scored_rows[index][field] for index in indices]
                require_near_balance(
                    values,
                    LETTERS,
                    f"{context} {TYPE_NAMES[trial_type]} {field}",
                )
        for field in ("CenterLetter", "FlankerLetter"):
            require_near_balance(
                [row[field] for row in scored_rows],
                LETTERS,
                f"{context} overall {field}",
            )
        return

    variable_cells = (
        (0, "CenterLetter"),
        (0, "FlankerLetter"),
        (1, "CenterLetter"),
        (3, "FlankerLetter"),
    )
    non_targets = tuple(LETTERS[index] for index in NON_TARGET_LETTERS)
    for trial_type, field in variable_cells:
        values = [
            row[field]
            for row, value in zip(scored_rows, types)
            if value == trial_type
        ]
        require_near_balance(
            values,
            non_targets,
            f"{context} {TYPE_NAMES[trial_type]} {field}",
        )
    for field in ("CenterLetter", "FlankerLetter"):
        values = [row[field] for row in scored_rows]
        counts = Counter(values)
        require(
            counts["H"] == len(scored_rows) // 2,
            f"{context}: overall {field} H count is not 50%",
        )
        require_near_balance(
            [value for value in values if value != "H"],
            non_targets,
            f"{context} overall non-H {field}",
        )


def read_and_validate_csv(key: ListKey, path: Path) -> ValidatedList:
    context = path.name
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.reader(handle))
    require(bool(raw_rows), f"{context}: file is empty")
    require(
        tuple(raw_rows[0]) == HEADER,
        f"{context}: header is not exactly {HEADER}",
    )
    require(
        all(len(row) == len(HEADER) for row in raw_rows[1:]),
        f"{context}: malformed or blank row",
    )
    rows = [dict(zip(HEADER, row)) for row in raw_rows[1:]]

    scored_offset = 1 if key.task == "1back" else 0
    expected_displayed = key.n_scored + scored_offset
    require(
        len(rows) == expected_displayed,
        f"{context}: expected {expected_displayed} data rows, found {len(rows)}",
    )
    start_number = 0 if scored_offset else 1
    expected_numbers = [
        str(value) for value in range(start_number, key.n_scored + 1)
    ]
    require(
        [row["TrialNumber"] for row in rows] == expected_numbers,
        f"{context}: TrialNumber sequence is incorrect",
    )
    require(
        all(
            row["CenterLetter"] in LETTERS
            and row["FlankerLetter"] in LETTERS
            for row in rows
        ),
        f"{context}: invalid letter identity",
    )

    if scored_offset:
        burn = rows[0]
        require(
            burn["CenterLetter"] != burn["FlankerLetter"],
            f"{context}: trial 0 center/flanker must differ",
        )
        require(
            burn["PerceptualCongruency"] == "no",
            f"{context}: trial 0 PerceptualCongruency must be no",
        )
        require(
            burn["BehavioralCongruency"] == ""
            and burn["CorrectResponse"] == ""
            and burn["ConditionType"] == "",
            f"{context}: trial 0 scored fields must be blank",
        )

    scored_rows = rows[scored_offset:]
    types: list[int] = []
    for scored_index, row in enumerate(scored_rows):
        display_index = scored_index + scored_offset
        categorical = tuple(row[field] for field in BINARY_FIELDS)
        require(
            all(value in {"yes", "no"} for value in categorical),
            f"{context}: scored categorical value is not lowercase yes/no",
        )
        require(
            row["ConditionType"] in TYPE_NAMES,
            f"{context}: invalid ConditionType at TrialNumber {scored_index + 1}",
        )
        trial_type = TYPE_NAMES.index(row["ConditionType"])
        require(
            categorical == TYPE_FIELDS[key.task][trial_type],
            f"{context}: fields do not match {row['ConditionType']} at "
            f"TrialNumber {scored_index + 1}",
        )

        center = row["CenterLetter"]
        flanker = row["FlankerLetter"]
        perceptual_yes = center == flanker
        if key.task == "0back":
            correct_yes = center == "H"
            behavioral_yes = (center == "H") == (flanker == "H")
        else:
            previous_center = rows[display_index - 1]["CenterLetter"]
            correct_yes = center == previous_center
            behavioral_yes = correct_yes == perceptual_yes
        derived = (
            "yes" if perceptual_yes else "no",
            "yes" if behavioral_yes else "no",
            "yes" if correct_yes else "no",
        )
        require(
            categorical == derived,
            f"{context}: derived fields are incorrect at TrialNumber "
            f"{scored_index + 1}",
        )
        types.append(trial_type)

    type_tuple = tuple(types)
    validate_type_order(type_tuple, key.task, key.block, context)
    validate_task_rules(key.task, rows, scored_offset, types, context)
    validate_identity_balance(key.task, scored_rows, types, context)

    binary_max_runs = {
        field: max_run([row[field] for row in scored_rows])
        for field in BINARY_FIELDS
    }
    all_centers = [row["CenterLetter"] for row in rows]
    all_flankers = [row["FlankerLetter"] for row in rows]
    raw_max_runs = {
        "CenterLetter": max_run(all_centers),
        "FlankerLetter": max_run(all_flankers),
        "LetterPair": max_run(list(zip(all_centers, all_flankers))),
    }
    require(
        raw_max_runs["CenterLetter"] <= 4,
        f"{context}: center identity run exceeds 4",
    )
    require(
        raw_max_runs["FlankerLetter"] <= 4,
        f"{context}: flanker identity run exceeds 4",
    )
    require(
        raw_max_runs["LetterPair"] <= 3,
        f"{context}: identical-pair run exceeds 3",
    )

    complete_rows = tuple(
        tuple(row[field] for field in HEADER[1:]) for row in scored_rows
    )
    require(
        repeated_divisor_period(complete_rows) == key.n_scored,
        f"{context}: complete rows repeat a shorter motif",
    )
    return ValidatedList(
        key=key,
        rows=tuple(tuple(row[field] for field in HEADER) for row in rows),
        types=type_tuple,
        complete_rows=complete_rows,
        binary_max_runs=binary_max_runs,
        raw_max_runs=raw_max_runs,
    )


def validate_cross_version_groups(
    validated: Sequence[ValidatedList],
) -> dict[tuple[str, str, int, int], tuple[int, int, int]]:
    distances: dict[tuple[str, str, int, int], tuple[int, int, int]] = {}
    for task in TASKS:
        for block in BLOCKS:
            group = sorted(
                (
                    value
                    for value in validated
                    if value.key.task == task and value.key.block == block
                ),
                key=lambda value: value.key.version,
            )
            require(
                [value.key.version for value in group] == list(VERSIONS),
                f"{task} {block}: missing ListVersion",
            )
            for left, right in itertools.combinations(group, 2):
                threshold = left.key.n_scored // 2
                type_distance = hamming(left.types, right.types)
                response_distance = hamming(
                    tuple(row[4] for row in left.complete_rows),
                    tuple(row[4] for row in right.complete_rows),
                )
                row_distance = hamming(left.complete_rows, right.complete_rows)
                context = (
                    f"{task} {block} ListVersion{left.key.version}/{right.key.version}"
                )
                require(
                    type_distance >= threshold,
                    f"{context}: type Hamming distance {type_distance} < {threshold}",
                )
                require(
                    row_distance >= threshold,
                    f"{context}: row Hamming distance {row_distance} < {threshold}",
                )
                require(
                    response_distance >= threshold,
                    f"{context}: CorrectResponse Hamming distance "
                    f"{response_distance} < {threshold}",
                )
                require(
                    not dihedral_equivalent(left.types, right.types),
                    f"{context}: type sequences are rotation/reversal equivalent",
                )
                require(
                    not dihedral_equivalent(left.complete_rows, right.complete_rows),
                    f"{context}: complete rows are rotation/reversal equivalent",
                )
                distances[(task, block, left.key.version, right.key.version)] = (
                    type_distance,
                    response_distance,
                    row_distance,
                )

    for version in VERSIONS:
        zero = next(
            item
            for item in validated
            if item.key == ListKey("0back", "main", version)
        )
        one = next(
            item
            for item in validated
            if item.key == ListKey("1back", "main", version)
        )
        require(
            zero.types == one.types,
            f"ListVersion{version}: main ConditionType orders differ across tasks",
        )
    return distances


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def html_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th scope=\"col\">{escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def display_field(field: str) -> str:
    return {
        "PerceptualCongruency": "Perceptual",
        "BehavioralCongruency": "Behavioral",
        "CorrectResponse": "Correct response",
    }[field]


def render_main_list(
    validated: ValidatedList, path: Path
) -> str:
    rows = rows_as_dicts(validated.rows)
    scored_offset = 1 if validated.key.task == "1back" else 0
    scored_rows = rows[scored_offset:]

    quarter_rows: list[list[object]] = []
    for quarter in range(4):
        counts = Counter(validated.types[quarter * 40 : (quarter + 1) * 40])
        quarter_rows.append(
            [
                f"Q{quarter + 1} ({quarter * 40 + 1}–{(quarter + 1) * 40})",
                *[counts[index] for index in range(4)],
            ]
        )
    total_counts = Counter(validated.types)
    quarter_rows.append(
        ["Total", *[total_counts[index] for index in range(4)]]
    )

    identity_run_rows: list[list[object]] = []
    for field in ("CenterLetter", "FlankerLetter"):
        values = [row[field] for row in rows]
        episodes = exact_run_episodes(values)
        identity_run_rows.append(
            [
                "Center" if field == "CenterLetter" else "Flanker",
                *[
                    sum(
                        count
                        for (_, episode_length), count in episodes.items()
                        if episode_length == length
                    )
                    for length in (2, 3, 4, 5)
                ],
                max_run(values),
            ]
        )

    binary_run_rows: list[list[object]] = []
    for field in BINARY_FIELDS:
        values = [row[field] for row in scored_rows]
        episodes = exact_run_episodes(values)
        for value in ("yes", "no"):
            binary_run_rows.append(
                [
                    display_field(field),
                    value,
                    *[episodes[(value, length)] for length in (2, 3, 4)],
                ]
            )
        binary_run_rows.append(
            [
                display_field(field),
                "total",
                *[
                    sum(
                        episodes[(value, length)] for value in ("yes", "no")
                    )
                    for length in (2, 3, 4)
                ],
            ]
        )

    letter_rows = []
    for field in ("CenterLetter", "FlankerLetter"):
        counts = Counter(row[field] for row in scored_rows)
        letter_rows.append(
            [
                "Center" if field == "CenterLetter" else "Flanker",
                *[counts[letter] for letter in LETTERS],
                len(scored_rows),
            ]
        )

    transition_rows = []
    transition_order = (
        ("yes", "yes"),
        ("yes", "no"),
        ("no", "yes"),
        ("no", "no"),
    )
    for field in BINARY_FIELDS:
        counts = transition_counts([row[field] for row in scored_rows])
        transition_rows.append(
            [display_field(field), *[counts[pair] for pair in transition_order]]
        )

    binary_counts = {
        field: Counter(row[field] for row in scored_rows) for field in BINARY_FIELDS
    }
    burn_note = (
        " Identity runs include response-free trial 0."
        if validated.key.task == "1back"
        else ""
    )
    relative_path = path.relative_to(ROOT).as_posix()
    return f"""
      <article class="list-card" id="{validated.key.task}-v{validated.key.version}">
        <header class="list-header">
          <div>
            <p class="eyebrow">ListVersion {validated.key.version}</p>
            <h3>{'0-back' if validated.key.task == '0back' else '1-back'} main</h3>
          </div>
          <span class="pass">Validated</span>
        </header>
        <div class="summary-grid">
          <div><span>Scored trials</span><strong>{validated.key.n_scored}</strong></div>
          <div><span>Conditions</span><strong>40 each</strong></div>
          <div><span>Perceptual</span><strong>{binary_counts['PerceptualCongruency']['yes']} / {binary_counts['PerceptualCongruency']['no']}</strong></div>
          <div><span>Behavioral</span><strong>{binary_counts['BehavioralCongruency']['yes']} / {binary_counts['BehavioralCongruency']['no']}</strong></div>
          <div><span>Response</span><strong>{binary_counts['CorrectResponse']['yes']} / {binary_counts['CorrectResponse']['no']}</strong></div>
          <div><span>CSV hash</span><strong class="mono">{sha256(path)}</strong></div>
        </div>
        <p class="file-link"><a href="{escape(relative_path)}">{escape(path.name)}</a></p>

        <div class="metric-grid">
          <section class="metric span-2">
            <h4>Condition distribution by quarter</h4>
            <p>Each quarter is constrained to 8–12 occurrences of every condition.</p>
            {html_table(('Quarter', *TYPE_NAMES), quarter_rows)}
          </section>

          <section class="metric">
            <h4>Exact letter-identity runs</h4>
            <p>Maximal episodes only; a length-3 episode is not also counted as length 2.{burn_note}</p>
            {html_table(('Position', 'Run 2', 'Run 3', 'Run 4', 'Run 5', 'Maximum'), identity_run_rows)}
          </section>

          <section class="metric">
            <h4>Scored letter counts</h4>
            <p>Trial 0 is excluded from 1-back identity quotas.</p>
            {html_table(('Position', *LETTERS, 'Total'), letter_rows)}
          </section>

          <section class="metric span-2">
            <h4>Exact binary-field runs</h4>
            <p>Runs are detected across the complete scored sequence and reported by repeated value and pooled total.</p>
            {html_table(('Field', 'Value', 'Run 2', 'Run 3', 'Run 4'), binary_run_rows)}
          </section>

          <section class="metric span-2">
            <h4>First-order transitions</h4>
            <p>No wrap-around transition is counted. Each row must be a permutation of 39, 40, 40, 40.</p>
            {html_table(('Field', 'yes→yes', 'yes→no', 'no→yes', 'no→no'), transition_rows)}
          </section>
        </div>
      </article>
    """.rstrip() + "\n"


def write_report(
    validated: Sequence[ValidatedList],
    paths: dict[ListKey, Path],
    distances: dict[tuple[str, str, int, int], tuple[int, int, int]],
) -> None:
    main_sections = []
    for task in TASKS:
        cards = "".join(
            render_main_list(
                next(
                    item
                    for item in validated
                    if item.key == ListKey(task, "main", version)
                ),
                paths[ListKey(task, "main", version)],
            )
            for version in VERSIONS
        )
        main_sections.append(
            f"""
            <section class="task-section" id="{'zero-back' if task == '0back' else 'one-back'}-properties">
              <div class="section-heading">
                <p class="eyebrow">Generated properties</p>
                <h2>{'0-back' if task == '0back' else '1-back'} main lists</h2>
                <p>Diagnostics are shown separately for all three fixed versions.</p>
              </div>
              {cards}
            </section>
            """
        )

    practice_rows = []
    for version in VERSIONS:
        for task in TASKS:
            key = ListKey(task, "practice", version)
            item = next(value for value in validated if value.key == key)
            displayed = len(item.rows)
            practice_rows.append(
                [
                    key.filename,
                    version,
                    "0-back" if task == "0back" else "1-back",
                    displayed,
                    key.n_scored,
                    "5 / 5 / 5 / 5",
                    max(item.binary_max_runs.values()),
                    max(item.raw_max_runs["CenterLetter"], item.raw_max_runs["FlankerLetter"]),
                    sha256(paths[key]),
                    "PASS",
                ]
            )

    distance_rows = []
    for (task, block, left_version, right_version), (
        type_distance, response_distance, row_distance
    ) in distances.items():
        threshold = 10 if block == "practice" else 80
        distance_rows.append(
            [
                "0-back" if task == "0back" else "1-back",
                block,
                f"{left_version} / {right_version}",
                type_distance,
                response_distance,
                row_distance,
                f"≥ {threshold}",
                "PASS",
            ]
        )

    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>N-back Flanker Stimulus Lists</title>
  <style>
    :root {
      --ink: #17212b;
      --muted: #5d6b78;
      --paper: #f4f6f2;
      --card: #ffffff;
      --line: #d8dfd8;
      --navy: #17324d;
      --teal: #0b7269;
      --teal-soft: #e4f3ef;
      --amber: #aa6417;
      --amber-soft: #fff2dd;
      --shadow: 0 12px 34px rgba(23, 50, 77, 0.08);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }
    a { color: var(--teal); }
    code, .mono {
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 0.92em;
    }
    .page { width: min(1180px, calc(100% - 40px)); margin: 0 auto; }
    .hero {
      color: white;
      background:
        radial-gradient(circle at 85% 10%, rgba(40, 173, 153, 0.28), transparent 34%),
        linear-gradient(135deg, #122a40, #173f58 62%, #0b625e);
      padding: 70px 0 54px;
    }
    .hero h1 {
      max-width: 850px;
      margin: 8px 0 18px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(2.4rem, 5vw, 4.5rem);
      line-height: 1.04;
      letter-spacing: -0.035em;
    }
    .hero .lede { max-width: 780px; font-size: 1.12rem; color: #dfecea; }
    .eyebrow {
      margin: 0;
      color: var(--teal);
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.13em;
      text-transform: uppercase;
    }
    .hero .eyebrow { color: #8fe0d2; }
    .status-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 26px; }
    .status-row span {
      padding: 7px 11px;
      border: 1px solid rgba(255,255,255,0.28);
      border-radius: 999px;
      background: rgba(255,255,255,0.09);
      font-size: 0.84rem;
      font-weight: 700;
    }
    .toc {
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(255,255,255,0.94);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(12px);
    }
    .toc .page { display: flex; gap: 22px; overflow-x: auto; padding: 12px 0; }
    .toc a { color: var(--navy); font-size: 0.86rem; font-weight: 750; text-decoration: none; white-space: nowrap; }
    main { padding: 54px 0 80px; }
    .section-heading { max-width: 800px; margin-bottom: 24px; }
    h2 {
      margin: 5px 0 10px;
      color: var(--navy);
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(1.8rem, 3vw, 2.7rem);
      line-height: 1.15;
    }
    h3 { margin: 2px 0 0; color: var(--navy); font-size: 1.45rem; }
    h4 { margin: 0 0 7px; color: var(--navy); font-size: 1rem; }
    p { margin: 0 0 12px; }
    .concept-grid, .definition-grid, .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }
    .concept, .definition, .metric, .list-card, .panel {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--card);
      box-shadow: var(--shadow);
    }
    .concept, .definition, .panel { padding: 24px; }
    .concept .formula {
      margin: 16px 0;
      padding: 13px 15px;
      border-left: 4px solid var(--teal);
      background: var(--teal-soft);
      font-family: "SFMono-Regular", Consolas, monospace;
      font-weight: 700;
    }
    .callout {
      margin-top: 18px;
      padding: 18px 20px;
      border: 1px solid #efcf9e;
      border-radius: 12px;
      background: var(--amber-soft);
    }
    .task-section, .report-section { margin-top: 64px; }
    .list-card { margin-top: 24px; padding: 26px; }
    .list-header { display: flex; justify-content: space-between; align-items: center; gap: 20px; }
    .pass {
      padding: 7px 11px;
      border-radius: 999px;
      color: #07564f;
      background: var(--teal-soft);
      font-size: 0.78rem;
      font-weight: 850;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 1px;
      margin: 22px 0 12px;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--line);
    }
    .summary-grid div { padding: 13px; background: #fbfcfa; }
    .summary-grid span { display: block; color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .summary-grid strong { display: block; margin-top: 3px; font-size: 0.94rem; }
    .file-link { margin-bottom: 20px; font-size: 0.86rem; }
    .metric { padding: 19px; box-shadow: none; }
    .metric > p { color: var(--muted); font-size: 0.86rem; }
    .span-2 { grid-column: 1 / -1; }
    .table-wrap { overflow-x: auto; margin-top: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.86rem; font-variant-numeric: tabular-nums; }
    th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
    th { color: var(--navy); background: #f0f4f2; font-size: 0.76rem; letter-spacing: 0.03em; }
    th:first-child, td:first-child { text-align: left; }
    tbody tr:last-child td { border-bottom: 0; }
    tbody tr:nth-child(even) { background: #fafbf9; }
    .rule-list { counter-reset: rules; display: grid; gap: 12px; padding: 0; list-style: none; }
    .rule-list li { position: relative; padding: 16px 18px 16px 54px; border: 1px solid var(--line); border-radius: 12px; background: white; }
    .rule-list li::before {
      counter-increment: rules;
      content: counter(rules);
      position: absolute;
      left: 16px;
      top: 14px;
      width: 25px;
      height: 25px;
      display: grid;
      place-items: center;
      border-radius: 50%;
      color: white;
      background: var(--navy);
      font-size: 0.75rem;
      font-weight: 800;
    }
    .rule-list strong { color: var(--navy); }
    footer { padding: 30px 0 45px; color: var(--muted); border-top: 1px solid var(--line); font-size: 0.85rem; }
    @media (max-width: 900px) {
      .summary-grid { grid-template-columns: repeat(3, 1fr); }
    }
    @media (max-width: 700px) {
      .page { width: min(100% - 24px, 1180px); }
      .hero { padding-top: 48px; }
      .concept-grid, .definition-grid, .metric-grid { grid-template-columns: 1fr; }
      .span-2 { grid-column: auto; }
      .summary-grid { grid-template-columns: repeat(2, 1fr); }
      .list-card { padding: 17px; }
    }
    @media print {
      .toc { position: static; }
      .list-card, .concept, .definition, .metric, .panel { box-shadow: none; break-inside: avoid; }
      a { color: inherit; text-decoration: none; }
    }
  </style>
</head>
<body>
  <header class="hero">
    <div class="page">
      <p class="eyebrow">Final stimulus framework</p>
      <h1>N-back Flanker Stimulus Lists</h1>
      <p class="lede">A single, task-specific account of perceptual and behavioral congruency, the constraints used to generate three fixed list versions, and the observed properties of every main list.</p>
      <div class="status-row">
        <span>12 CSVs validated</span>
        <span>3 fixed versions</span>
        <span>160 scored main trials per task</span>
        <span>Deterministic output</span>
      </div>
    </div>
  </header>

  <nav class="toc" aria-label="Report sections">
    <div class="page">
      <a href="#constructs">Constructs</a>
      <a href="#conditions">Conditions</a>
      <a href="#constraints">Constraints</a>
      <a href="#validation">Validation</a>
      <a href="#zero-back-properties">0-back lists</a>
      <a href="#one-back-properties">1-back lists</a>
      <a href="#practice">Practice</a>
      <a href="#versions">Versions</a>
    </div>
  </nav>

  <main class="page">
    <section id="constructs">
      <div class="section-heading">
        <p class="eyebrow">Fundamental definitions — not list constraints</p>
        <h2>Two meanings of congruency</h2>
        <p>Let <code>C<sub>t</sub></code> and <code>F<sub>t</sub></code> denote the current center and flanker letters. Previous-trial letters are <code>C<sub>t−1</sub></code> and <code>F<sub>t−1</sub></code>.</p>
      </div>
      <div class="concept-grid">
        <article class="concept">
          <p class="eyebrow">Perceptual congruency</p>
          <h3>Do the current letters match?</h3>
          <div class="formula">P = (F<sub>t</sub> = C<sub>t</sub>)</div>
          <p>This definition is identical in 0-back and 1-back. It concerns simultaneous visual identity only.</p>
        </article>
        <article class="concept">
          <p class="eyebrow">Behavioral congruency</p>
          <h3>Does irrelevant evidence support the task response?</h3>
          <p>The relevant behavioral comparison changes with the task. Behavioral congruency therefore has a task-specific derivation even though its final CSV label remains <code>yes</code> or <code>no</code>.</p>
        </article>
      </div>
      <div class="definition-grid" style="margin-top:18px">
        <article class="definition">
          <p class="eyebrow">0-back behavioral rule</p>
          <h3>H versus non-H response mapping</h3>
          <p>Define <code>R<sub>0</sub>(X) = (X = H)</code>. Behavioral congruency is:</p>
          <p class="formula"><code>R<sub>0</sub>(C<sub>t</sub>) = R<sub>0</sub>(F<sub>t</sub>)</code></p>
          <p>The center and flanker are behaviorally congruent when both map to YES or both map to NO under the fixed-target rule.</p>
        </article>
        <article class="definition">
          <p class="eyebrow">1-back behavioral rule</p>
          <h3>Match-status compatibility</h3>
          <p>Define the task match <code>T = (C<sub>t</sub> = C<sub>t−1</sub>)</code>. Behavioral congruency is:</p>
          <p class="formula"><code>T = P</code></p>
          <p><code>F<sub>t</sub> = F<sub>t−1</sub></code> can strengthen match evidence in selected conditions, but it is a construction rule rather than the definition of behavioral congruency.</p>
        </article>
      </div>
      <div class="callout"><strong>Important:</strong> behavioral congruency is derived from task-relevant response compatibility. It is not an independently crossed third factor, and no difficulty category is assigned.</div>
    </section>

    <section class="report-section" id="conditions">
      <div class="section-heading">
        <p class="eyebrow">Operational definitions</p>
        <h2>The four selected conditions</h2>
        <p>A denotes a correct NO response; B denotes a correct YES response. Suffix 1 is behaviorally congruent and suffix 2 is behaviorally incongruent.</p>
      </div>
      <article class="panel">
        <h3>0-back: target letter H</h3>
        <div class="table-wrap"><table>
          <thead><tr><th>Type</th><th>Correct</th><th>Center and flanker rules</th><th>Perceptual</th><th>Behavioral</th></tr></thead>
          <tbody>
            <tr><td>A1</td><td>no</td><td><code>C<sub>t</sub> ≠ H; F<sub>t</sub> = C<sub>t</sub></code></td><td>yes</td><td>yes</td></tr>
            <tr><td>A2</td><td>no</td><td><code>C<sub>t</sub> ≠ H; F<sub>t</sub> = H</code></td><td>no</td><td>no</td></tr>
            <tr><td>B1</td><td>yes</td><td><code>C<sub>t</sub> = F<sub>t</sub> = H</code></td><td>yes</td><td>yes</td></tr>
            <tr><td>B2</td><td>yes</td><td><code>C<sub>t</sub> = H; F<sub>t</sub> ≠ H</code></td><td>no</td><td>no</td></tr>
          </tbody>
        </table></div>
        <p class="callout">The perceptually incongruent but behaviorally congruent non-H/non-H case is deliberately omitted. The selected 0-back conditions occupy only the perceptual/behavioral diagonal.</p>
      </article>
      <article class="panel" style="margin-top:18px">
        <h3>1-back: current versus previous center</h3>
        <div class="table-wrap"><table>
          <thead><tr><th>Type</th><th>Correct</th><th>Center and flanker rules</th><th>Perceptual</th><th>Behavioral</th></tr></thead>
          <tbody>
            <tr><td>A1</td><td>no</td><td><code>C<sub>t</sub> ≠ C<sub>t−1</sub>; F<sub>t</sub> ∉ {C<sub>t</sub>, C<sub>t−1</sub>, F<sub>t−1</sub>}</code></td><td>no</td><td>yes</td></tr>
            <tr><td>A2</td><td>no</td><td><code>C<sub>t</sub> ≠ C<sub>t−1</sub>; F<sub>t</sub> = C<sub>t</sub> = F<sub>t−1</sub></code></td><td>yes</td><td>no</td></tr>
            <tr><td>B1</td><td>yes</td><td><code>C<sub>t</sub> = C<sub>t−1</sub>; F<sub>t</sub> = C<sub>t</sub></code></td><td>yes</td><td>yes</td></tr>
            <tr><td>B2</td><td>yes</td><td><code>C<sub>t</sub> = C<sub>t−1</sub>; F<sub>t</sub> ∉ {C<sub>t</sub>, F<sub>t−1</sub>}</code></td><td>no</td><td>no</td></tr>
          </tbody>
        </table></div>
      </article>
    </section>

    <section class="report-section" id="constraints">
      <div class="section-heading">
        <p class="eyebrow">Generation requirements</p>
        <h2>Actual list constraints</h2>
        <p>These requirements govern the composition and ordering of generated rows; unlike the preceding construct definitions, they can pass or fail.</p>
      </div>
      <ol class="rule-list">
        <li><strong>Output contract.</strong> CSV columns are exactly <code>TrialNumber, CenterLetter, FlankerLetter, PerceptualCongruency, BehavioralCongruency, CorrectResponse, ConditionType</code>. Practice/main contain 20/160 scored trials. Each 1-back file adds response-free trial 0.</li>
        <li><strong>Condition composition.</strong> Practice contains 5 and main contains 40 of each A1/A2/B1/B2. Consequently, each binary field is exactly 50% yes and 50% no over scored trials.</li>
        <li><strong>Main-quarter structure.</strong> Trials 1–40, 41–80, 81–120, and 121–160 each contain 8–12 of every condition. Within a ListVersion, 0-back and 1-back share the same main ConditionType order.</li>
        <li><strong>Letter-identity runs.</strong> Across displayed rows, center and flanker identity runs are capped at 4 and identical center/flanker-pair runs at 3. In 1-back, a run of three consecutive YES responses requires four identical centers.</li>
        <li><strong>Congruency sequences.</strong> PerceptualCongruency and BehavioralCongruency each have maximum run length 3 across scored trials, including across quarter boundaries.</li>
        <li><strong>Response sequence.</strong> CorrectResponse independently has maximum run length 3 across scored trials. In every main list, YES and NO have equal exact run-3 counts, with 2–4 per value and fewer than 10 pooled; their exact run-2 counts differ by no more than 5. The three main ListVersions use different run-2 count pairs.</li>
        <li><strong>First-order transitions.</strong> For every binary field, yes→yes, yes→no, no→yes, and no→no differ by at most 1: sorted counts are [4,5,5,5] in practice and [39,40,40,40] in main.</li>
        <li><strong>Letter counts.</strong> Identities are balanced as evenly as mathematically possible both within each legally variable condition/position cell and overall. In 0-back, H necessarily occupies 50% of both positions; S/C/F split the remainder. In 1-back main, every condition/position cell contains exactly 10 of each letter.</li>
        <li><strong>Distinct versions.</strong> Each pair of versions differs at ≥50% of scored positions in ConditionType, CorrectResponse, and complete-row content; they are not cyclic rotations, reversed rotations, or repetitions of a shorter divisor-sized motif.</li>
      </ol>
    </section>

    <section class="report-section" id="validation">
      <div class="section-heading">
        <p class="eyebrow">Verification rules</p>
        <h2>How validation is scoped</h2>
      </div>
      <div class="definition-grid">
        <article class="definition"><h3>Trial 0</h3><p>Excluded from scored condition counts, quarters, binary runs, transitions, letter quotas, and cross-version comparisons. Included in displayed center/flanker run checks and used as the previous state for 1-back trial 1.</p></article>
        <article class="definition"><h3>Read-back validation</h3><p>Every CSV is read and checked against its header, row numbering, logical task rules, derived labels, all sequence/count constraints, and version-level requirements using explicit exceptions. Existing CSVs must match the deterministic output and are preserved byte for byte; only missing files are written.</p></article>
      </div>
    </section>
"""
    document += "".join(main_sections)
    document += f"""
    <section class="report-section" id="practice">
      <div class="section-heading">
        <p class="eyebrow">Concise audit</p>
        <h2>Practice lists</h2>
        <p>Practice lists receive the same applicable logical, balance, run, transition, identity, and distinctness checks; detailed diagnostics are reserved for main lists.</p>
      </div>
      <article class="panel">
        {html_table(('File', 'Version', 'Task', 'Displayed', 'Scored', 'A1/A2/B1/B2', 'Max binary run', 'Max letter run', 'CSV hash', 'Status'), practice_rows)}
      </article>
    </section>

    <section class="report-section" id="versions">
      <div class="section-heading">
        <p class="eyebrow">Cross-version validation</p>
        <h2>Materially distinct fixed lists</h2>
        <p>Hamming distances exclude TrialNumber and the 1-back trial 0. Main ConditionType orders are matched across tasks within each version, while versions 1, 2, and 3 remain materially different in every pairwise comparison.</p>
      </div>
      <article class="panel">
        {html_table(('Task', 'Block', 'Versions', 'Type distance', 'Response distance', 'Complete-row distance', 'Required', 'Status'), distance_rows)}
      </article>
    </section>
  </main>

  <footer>
    <div class="page">Generated deterministically by <code>generate_stimulus_lists.py</code> after all CSV read-back validations passed.</div>
  </footer>
</body>
</html>
"""
    REPORT_PATH.write_text(document, encoding="utf-8")


def main() -> None:
    generated = build_all_lists()
    paths = write_lists(generated)
    validated = [
        read_and_validate_csv(key, path) for key, path in paths.items()
    ]
    distances = validate_cross_version_groups(validated)
    write_report(validated, paths, distances)

    print(f"Generated and validated {len(paths)} CSV files in {OUTPUT_DIRECTORY}")
    for key in sorted(
        paths, key=lambda value: (value.version, value.task, value.block)
    ):
        item = next(value for value in validated if value.key == key)
        print(
            f"  {key.filename}: sha256={sha256(paths[key])}, "
            f"binary_max_runs={item.binary_max_runs}, "
            f"raw_max_runs={item.raw_max_runs}"
        )
    for (
        task,
        block,
        left_version,
        right_version,
    ), (type_distance, response_distance, row_distance) in distances.items():
        print(
            f"  {task} {block}: ListVersion{left_version}/{right_version} Hamming distance "
            f"types={type_distance}, responses={response_distance}, "
            f"complete_rows={row_distance}"
        )
    print(f"Generated report: {REPORT_PATH} (sha256={sha256(REPORT_PATH)})")


if __name__ == "__main__":
    main()
