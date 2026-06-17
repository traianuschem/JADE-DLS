"""
Data Provenance Record for JADE-DLS
PROV-inspired JSON schema for FAIR-compliant tracking of DLS analysis sessions.

Schema: agent → input entities (SHA-256) → processing activities (DAG) → output catalog.
Each record carries a UUID4 record_id that is embedded in every exported artifact
so reports can be unambiguously traced back to their provenance record.
"""

import hashlib
import json
import platform
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_URI = "https://jade-dls.de/schema/provenance/v1.0"

_STEP_TYPE_MAP: Dict[str, str] = {
    "function": "analysis",
    "filter": "filter",
    "refinement": "refinement",
    # "custom" is handled dynamically by _infer_activity_type()
}

# Keywords used to infer the semantic type of "custom" steps from their name
_LOAD_KEYWORDS = ("load", "extract", "import", "read")
_FILTER_KEYWORDS = ("filter", "exclude", "remove")
_REFINE_KEYWORDS = ("refine", "refinement", "post-refine", "post-fit")


def _infer_activity_type(step_dict: Dict[str, Any]) -> str:
    """Return a provenance activity type for an AnalysisStep dict."""
    step_type = step_dict.get("step_type", "function")
    if step_type in _STEP_TYPE_MAP:
        return _STEP_TYPE_MAP[step_type]
    # For 'custom' steps (code-generated), infer from the step name
    name = step_dict.get("name", "").lower()
    if any(kw in name for kw in _REFINE_KEYWORDS):
        return "refinement"
    if any(kw in name for kw in _FILTER_KEYWORDS):
        return "filter"
    if any(kw in name for kw in _LOAD_KEYWORDS):
        return "data_loading"
    return "analysis"  # default for cumulant / NNLS / regularized / etc.


def compute_sha256(filepath: str) -> str:
    """Return SHA-256 hex digest for *filepath*, or empty string on I/O error."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, IOError):
        return ""


class ProvenanceRecord:
    """
    Provenance record for one JADE-DLS analysis session.

    Follows a simplified PROV-DM model:
      - *agent*: the software that performed the analysis
      - *input entities*: raw data files with SHA-256 fingerprints
      - *activities*: ordered processing steps forming a DAG (via ``used`` references)
      - *output catalog*: reports and derived files linked back to this record

    Usage::

        record = ProvenanceRecord(version="2.1.1")
        record.set_input_folder("/data/sample_run")
        record.add_input_entity("/data/sample_run/s001_90deg.asc",
                                metadata={"angle_deg": 90.0, "temperature_C": 25.0})
        act_id = record.add_activity_from_step(step.to_dict())
        record.add_output("report_pdf", "DLS_Report.pdf", "/out/DLS_Report.pdf")
        json_text = record.to_json()
    """

    def __init__(self, version: str = "unknown"):
        self.record_id: str = str(uuid.uuid4())
        self.created: str = datetime.now().isoformat()
        self.agent: Dict[str, str] = {
            "software": "JADE-DLS",
            "version": version,
            "platform": f"{platform.system()} {platform.version()}",
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}"
                f".{sys.version_info.micro}"
            ),
        }
        self._input_folder: Optional[str] = None
        self._input_entities: List[Dict[str, Any]] = []
        self._excluded_files: List[str] = []
        self._activities: List[Dict[str, Any]] = []
        self._outputs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Input population
    # ------------------------------------------------------------------

    def set_input_folder(self, folder: str) -> None:
        self._input_folder = str(folder)

    def add_input_entity(
        self,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a raw data file as an input entity with SHA-256 fingerprint."""
        p = Path(filepath)
        sha = compute_sha256(filepath)
        entity: Dict[str, Any] = {
            "id": f"sha256:{sha}" if sha else f"file:{p.name}",
            "filename": p.name,
            "path": str(filepath),
            "sha256": sha,
            "role": "raw_data",
        }
        if metadata:
            entity["metadata"] = {
                k: (float(v) if hasattr(v, 'item') else v)
                for k, v in metadata.items()
            }
        self._input_entities.append(entity)

    def mark_file_excluded(self, filename: str) -> None:
        """Record that a file was excluded from the analysis."""
        if filename not in self._excluded_files:
            self._excluded_files.append(filename)

    # ------------------------------------------------------------------
    # Activity tracking
    # ------------------------------------------------------------------

    def add_activity_from_step(self, step_dict: Dict[str, Any]) -> str:
        """
        Append an analysis activity from an AnalysisStep.to_dict() payload.

        Activities are chained: each new activity lists the previous one in
        its ``used`` field, forming a linear provenance DAG.

        Returns the new activity id (e.g. ``"act-003"``).
        """
        n = len(self._activities)
        act_id = f"act-{n + 1:03d}"
        prev_id = f"act-{n:03d}" if n > 0 else None

        # Sanitise params — drop anything that cannot be JSON-serialised
        raw_params = step_dict.get("params") or {}
        safe_params: Dict[str, Any] = {}
        for k, v in raw_params.items():
            try:
                json.dumps(v)
                safe_params[k] = v
            except (TypeError, ValueError):
                safe_params[k] = str(v)

        activity: Dict[str, Any] = {
            "id": act_id,
            "label": step_dict.get("name", "Unknown Step"),
            "type": _infer_activity_type(step_dict),
            "timestamp": step_dict.get("timestamp", datetime.now().isoformat()),
            "parameters": safe_params,
            "used": [prev_id] if prev_id else [],
        }
        self._activities.append(activity)
        return act_id

    def annotate_last_activity(self, results_summary: Dict[str, Any]) -> None:
        """Attach a key-result summary dict to the most recently added activity."""
        if self._activities:
            safe: Dict[str, Any] = {}
            for k, v in results_summary.items():
                try:
                    json.dumps(v)
                    safe[k] = v
                except (TypeError, ValueError):
                    safe[k] = str(v)
            self._activities[-1]["results_summary"] = safe

    # ------------------------------------------------------------------
    # Output catalog
    # ------------------------------------------------------------------

    def add_output(
        self,
        output_type: str,
        label: str,
        filepath: Optional[str] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register an exported artifact in the output catalog.

        If *filepath* is provided the file is hashed so the artifact can be
        verified later.  The ``record_id`` of this provenance record is
        embedded in the entry so any artifact can be traced back here.

        *extra_fields* is an optional dict of additional key/value pairs
        merged into the catalog entry (e.g. ``{"export_id": "<uuid>"}`` to
        cross-link an export-metadata JSON back to its provenance record).

        Returns the new output id (e.g. ``"out-002"``).
        """
        n = len(self._outputs)
        out_id = f"out-{n + 1:03d}"
        entry: Dict[str, Any] = {
            "id": out_id,
            "type": output_type,
            "label": label,
            "record_id": self.record_id,
            "wasGeneratedBy": self._activities[-1]["id"] if self._activities else None,
            "wasDerivedFrom": [e["id"] for e in self._input_entities],
            "timestamp": datetime.now().isoformat(),
        }
        if filepath:
            sha = compute_sha256(filepath)
            if sha:
                entry["sha256"] = sha
        if extra_fields:
            entry.update(extra_fields)
        self._outputs.append(entry)
        return out_id

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "$schema": SCHEMA_URI,
            "record_id": self.record_id,
            "created": self.created,
            "agent": self.agent,
            "input": {
                "data_folder": self._input_folder,
                "entities": self._input_entities,
                "excluded_files": self._excluded_files,
            },
            "processing": {
                "activities": self._activities,
            },
            "output": {
                "catalog": self._outputs,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_prov_json(self, indent: int = 2) -> str:
        """
        Serialise as W3C PROV-JSON (https://www.w3.org/TR/prov-json/).
        Produces a second, standards-compliant serialisation alongside the
        existing to_json() / to_dict() output, which is kept for GUI display.
        Compatible with the `prov` Python library and W3C PROV-Toolbox.
        """
        JADE = "https://jade-dls.de/ns/provenance#"

        def jade(local: str) -> str:
            return f"jade:{local}"

        def xsd_datetime(iso: str) -> dict:
            return {"$": iso, "type": "xsd:dateTime"}

        # --- prefix ---
        doc: Dict[str, Any] = {
            "prefix": {
                "xsd":  "http://www.w3.org/2001/XMLSchema#",
                "prov": "http://www.w3.org/ns/prov#",
                "jade": JADE,
            }
        }

        # --- agent ---
        agent_id = jade("agent-software")
        doc["agent"] = {
            agent_id: {
                "prov:type":           {"$": "prov:SoftwareAgent", "type": "xsd:QName"},
                jade("software"):      self.agent.get("software", "JADE-DLS"),
                jade("version"):       self.agent.get("version", "unknown"),
                jade("platform"):      self.agent.get("platform", ""),
                jade("pythonVersion"): self.agent.get("python_version", ""),
            }
        }

        # --- input entities ---
        doc["entity"] = {}
        for ent in self._input_entities:
            eid = jade(ent["id"].replace(":", "_"))
            doc["entity"][eid] = {
                "prov:type":      {"$": jade("RawDataFile"), "type": "xsd:QName"},
                jade("filename"): ent.get("filename", ""),
                jade("path"):     ent.get("path", ""),
                jade("sha256"):   ent.get("sha256", ""),
                jade("role"):     ent.get("role", "raw_data"),
            }
            for k, v in ent.get("metadata", {}).items():
                doc["entity"][eid][jade(k)] = v

        # --- output entities ---
        for out in self._outputs:
            oid = jade(out["id"])
            doc["entity"][oid] = {
                "prov:type":        {"$": jade("ExportedArtifact"), "type": "xsd:QName"},
                jade("outputType"): out.get("type", ""),
                jade("label"):      out.get("label", ""),
                jade("recordId"):   out.get("record_id", self.record_id),
            }
            if "sha256" in out:
                doc["entity"][oid][jade("sha256")] = out["sha256"]

        # --- activities ---
        doc["activity"] = {}
        for act in self._activities:
            aid = jade(act["id"])
            doc["activity"][aid] = {
                "prov:startedAtTime": xsd_datetime(act.get("timestamp",
                                                   datetime.now().isoformat())),
                "prov:type":          {"$": jade(act.get("type", "analysis").title()),
                                       "type": "xsd:QName"},
                jade("label"):        act.get("label", ""),
                jade("parameters"):   json.dumps(act.get("parameters", {})),
            }
            if "results_summary" in act:
                doc["activity"][aid][jade("resultsSummary")] = json.dumps(
                    act["results_summary"])

        # --- wasInformedBy (activity chain) ---
        doc["wasInformedBy"] = {}
        for act in self._activities:
            for prev_id in act.get("used", []):
                if prev_id:
                    bn = f"_:wib_{act['id']}_{prev_id}".replace("-", "_")
                    doc["wasInformedBy"][bn] = {
                        "prov:informed":  jade(act["id"]),
                        "prov:informant": jade(prev_id),
                    }

        # --- wasAssociatedWith (every activity ↔ software agent) ---
        doc["wasAssociatedWith"] = {}
        for i, act in enumerate(self._activities):
            bn = f"_:waw_{i + 1:03d}"
            doc["wasAssociatedWith"][bn] = {
                "prov:activity": jade(act["id"]),
                "prov:agent":    agent_id,
            }

        # --- wasGeneratedBy (output entity ← generating activity) ---
        doc["wasGeneratedBy"] = {}
        for out in self._outputs:
            gen_act = out.get("wasGeneratedBy")
            if gen_act:
                bn = f"_:wgb_{out['id']}".replace("-", "_")
                doc["wasGeneratedBy"][bn] = {
                    "prov:entity":   jade(out["id"]),
                    "prov:activity": jade(gen_act),
                    "prov:time":     xsd_datetime(out.get("timestamp",
                                                 datetime.now().isoformat())),
                }

        # --- wasDerivedFrom (output entity ← input entities) ---
        doc["wasDerivedFrom"] = {}
        for out in self._outputs:
            for src_id in out.get("wasDerivedFrom", []):
                bn = f"_:wdf_{out['id']}_{src_id}".replace(":", "_").replace("-", "_")
                doc["wasDerivedFrom"][bn] = {
                    "prov:generatedEntity": jade(out["id"]),
                    "prov:usedEntity":      jade(src_id.replace(":", "_")),
                }

        # --- JADE-specific metadata not covered by PROV-JSON ---
        doc[jade("sessionMetadata")] = {
            jade("recordId"):      self.record_id,
            jade("created"):       self.created,
            jade("dataFolder"):    self._input_folder or "",
            jade("excludedFiles"): self._excluded_files,
        }

        return json.dumps(doc, indent=indent, ensure_ascii=False)

    def export_prov_json_to_file(self, filepath: str) -> None:
        """Write the PROV-JSON serialisation to *filepath*."""
        p = Path(filepath)
        p.write_text(self.to_prov_json(), encoding="utf-8")
        self.add_output("provenance_prov_json", p.name, filepath)

    def export_to_file(self, filepath: str) -> None:
        """Write the provenance JSON to *filepath* and register it as an output."""
        p = Path(filepath)
        p.write_text(self.to_json(), encoding="utf-8")
        self.add_output("provenance_json", p.name, filepath)

    # ------------------------------------------------------------------
    # Builder from existing pipeline snapshot
    # ------------------------------------------------------------------

    @classmethod
    def from_pipeline(cls, pipeline, version: str = "unknown") -> "ProvenanceRecord":
        """Build a ProvenanceRecord from a TransparentPipeline snapshot."""
        record = cls(version=version)
        if "data_folder" in pipeline.data:
            record.set_input_folder(pipeline.data["data_folder"])
        for step in pipeline.steps:
            record.add_activity_from_step(step.to_dict())
        return record
