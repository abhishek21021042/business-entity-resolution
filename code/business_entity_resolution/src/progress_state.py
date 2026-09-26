"""
Progress State Management for Distributed Inference.
Records live metrics, speeds, ETA, and stage status into JSON for both CLI and Web UI.
"""

from pathlib import Path
import json
import time
from typing import Optional, Dict, Any


class ProgressTracker:
    def __init__(self, output_dir: Path, shard_id: int, num_shards: int, total_entities: int):
        self.output_dir = Path(output_dir)
        self.state_file = self.output_dir / "progress.json"
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.total_entities = total_entities
        self.start_time = time.time()
        self.last_update = time.time()
        
        self.data: Dict[str, Any] = {
            "status": "INITIALIZING",
            "shard_id": shard_id,
            "num_shards": num_shards,
            "shard_name": f"Part {shard_id + 1} of {num_shards}",
            "total_entities": total_entities,
            "entities_processed": 0,
            "percent": 0.0,
            "current_chunk": 0,
            "total_chunks": 0,
            "candidates_count": 0,
            "matches_count": 0,
            "speed_eps": 0.0,
            "elapsed_seconds": 0,
            "eta_seconds": 0,
            "stage": "Starting up...",
            "last_message": "Initializing shard runner...",
            "updated_at": time.time(),
        }
        self.save()

    def update_stage(self, stage: str, message: str = ""):
        self.data["stage"] = stage
        if message:
            self.data["last_message"] = message
        self.data["elapsed_seconds"] = int(time.time() - self.start_time)
        self.save()

    def update_chunk_progress(self, chunk_idx: int, total_chunks: int,
                              processed_entities: int,
                              new_candidates: int,
                              new_matches: int,
                              stage_msg: str = ""):
        now = time.time()
        elapsed = max(1, now - self.start_time)
        self.data["current_chunk"] = chunk_idx + 1
        self.data["total_chunks"] = total_chunks
        self.data["entities_processed"] = processed_entities
        self.data["candidates_count"] += new_candidates
        self.data["matches_count"] += new_matches
        
        pct = min(100.0, (processed_entities / max(1, self.total_entities)) * 100)
        self.data["percent"] = round(pct, 1)
        
        speed = processed_entities / elapsed
        self.data["speed_eps"] = round(speed, 1)
        
        rem_entities = max(0, self.total_entities - processed_entities)
        eta = rem_entities / max(1.0, speed)
        self.data["eta_seconds"] = int(eta)
        self.data["elapsed_seconds"] = int(elapsed)
        self.data["status"] = "RUNNING"
        if stage_msg:
            self.data["stage"] = stage_msg
            self.data["last_message"] = stage_msg
        
        self.save()
        self.render_cli_progress()

    def mark_complete(self, out_cand_file: Path, out_match_file: Path):
        self.data["status"] = "COMPLETED"
        self.data["percent"] = 100.0
        self.data["entities_processed"] = self.total_entities
        self.data["stage"] = "Shard finished successfully!"
        self.data["last_message"] = "Outputs saved. Ready for merge."
        self.data["elapsed_seconds"] = int(time.time() - self.start_time)
        self.data["eta_seconds"] = 0
        self.data["cand_file"] = str(out_cand_file)
        self.data["match_file"] = str(out_match_file)
        self.save()

    def mark_error(self, err_msg: str):
        self.data["status"] = "ERROR"
        self.data["stage"] = f"Failed: {err_msg}"
        self.data["last_message"] = err_msg
        self.save()

    def save(self):
        self.data["updated_at"] = time.time()
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            temp_file = self.output_dir / "progress.json.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            temp_file.replace(self.state_file)
        except Exception:
            pass

    def render_cli_progress(self):
        """Displays a clean ASCII progress bar on the terminal."""
        pct = self.data["percent"]
        bar_len = 25
        filled = int(bar_len * (pct / 100.0))
        bar = "█" * filled + "░" * (bar_len - filled)
        
        eta_m, eta_s = divmod(self.data["eta_seconds"], 60)
        elp_m, elp_s = divmod(self.data["elapsed_seconds"], 60)
        
        status_line = (
            f"\r[{bar}] {pct:5.1f}% | "
            f"Chunk {self.data['current_chunk']}/{self.data['total_chunks']} | "
            f"{self.data['entities_processed']:,}/{self.total_entities:,} S1 | "
            f"Speed: {self.data['speed_eps']:,.0f} ent/s | "
            f"Elapsed: {elp_m:02d}m{elp_s:02d}s | "
            f"ETA: {eta_m:02d}m{eta_s:02d}s"
        )
        print(status_line, end="", flush=True)


def read_progress_state(output_dir: Path) -> Dict[str, Any]:
    """Reads current progress state from file."""
    f = Path(output_dir) / "progress.json"
    if f.exists():
        try:
            with open(f, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            pass
    return {"status": "IDLE", "percent": 0.0, "stage": "Waiting to start..."}
