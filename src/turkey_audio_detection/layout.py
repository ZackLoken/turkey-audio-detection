"""Path and output layout helpers aligned to plan.md."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    project_root: Path
    run_id: str
    data_root: Path
    outputs_root: Path
    run_root: Path
    index_dir: Path
    birdnet_dir: Path
    queue_dir: Path
    clips_dir: Path
    manifests_dir: Path
    review_labels_dir: Path
    review_adjudication_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path, run_id: str) -> "RunLayout":
        data_root = project_root / "data"
        outputs_root = data_root / "_outputs"
        run_root = outputs_root / "runs" / run_id
        return cls(
            project_root=project_root,
            run_id=run_id,
            data_root=data_root,
            outputs_root=outputs_root,
            run_root=run_root,
            index_dir=run_root / "index",
            birdnet_dir=run_root / "birdnet",
            queue_dir=run_root / "queue",
            clips_dir=run_root / "clips",
            manifests_dir=run_root / "manifests",
            review_labels_dir=outputs_root / "review" / "labels",
            review_adjudication_dir=outputs_root / "review" / "adjudication",
        )

    def ensure_dirs(self) -> None:
        dirs = [
            self.index_dir,
            self.birdnet_dir,
            self.queue_dir,
            self.clips_dir,
            self.manifests_dir,
            self.review_labels_dir,
            self.review_adjudication_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


def find_aru_dirs(project_root: Path) -> list[Path]:
    data_root = project_root / "data"
    if not data_root.exists():
        return []
    return sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("ARU_")])


def validate_project_layout(project_root: Path) -> list[Path]:
    aru_dirs = find_aru_dirs(project_root)
    if not aru_dirs:
        raise ValueError(
            f"Expected one or more data/ARU_* folders under project root: {project_root}"
        )
    return aru_dirs
