#!/usr/bin/env python3
"""Basic Ralph - A research-plan-implement loop for coding agents.

OVERVIEW:
    Basic Ralph implements a three-phase workflow for AI-assisted software development:
    1. Research - Analyze the task and document findings
    2. Planning - Create an implementation plan based on research
    3. Implementation - Execute the plan to complete the task

THE RALPH LOOP:
    The core workflow (run command) executes tasks through multiple iterations:
    
    1. Creates a ticket for tracking the task
    2. RESEARCH PHASE: Calls opencode to research the task
       - Analyzes requirements, technologies, and challenges
       - Saves findings to .tickets/artifacts/<ticket_id>/research.md
       - Must output "READY_FOR_NEXT_TASK" to proceed
     
    3. PLANNING PHASE: Calls opencode to create implementation plan
       - Based on research findings
       - Saves plan to .tickets/artifacts/<ticket_id>/plan.md
       - Must output "READY_FOR_NEXT_TASK" to proceed
    
    4. IMPLEMENTATION PHASE: Calls opencode to implement the solution
       - Executes the plan from phase 3
       - Must output "COMPLETE" (configurable) to mark task done
    
    5. Repeats until completion or max iterations reached

USAGE - RUN COMMAND (The Ralph Loop):
    # Basic usage
    python basic_ralph.py run "Implement user authentication"
    
    # With options
    python basic_ralph.py run "Refactor database layer" \
        --model gpt-4 \
        --max-iterations 5 \
        --completion-promise "DONE"
    
    # From file
    python basic_ralph.py run --prompt-file task.txt
    
    # Multiple tasks
    python basic_ralph.py run -f task1.txt -f task2.txt "Third task"

USAGE - TICKET MANAGEMENT:
    # Create tickets
    python basic_ralph.py ticket create "Fix login bug" -d "Users can't log in"
    python basic_ralph.py ticket create "Add feature" -p 1 --type feature
    
    # Manage tickets
    python basic_ralph.py ticket ls                    # List all tickets
    python basic_ralph.py ticket ready                 # Show unblocked tickets
    python basic_ralph.py ticket blocked               # Show blocked tickets
    python basic_ralph.py ticket show <id>             # View ticket details
    python basic_ralph.py ticket start <id>            # Mark as in_progress
    python basic_ralph.py ticket close <id>            # Mark as closed
    
    # Dependencies
    python basic_ralph.py ticket dep <id> <dep_id>     # Add dependency
    python basic_ralph.py ticket undep <id> <dep_id>   # Remove dependency
    python basic_ralph.py ticket dep-tree <id>         # Show dependency tree

TICKET FILE FORMAT:
    Tickets are stored as Markdown files in .tickets/ with TOML frontmatter:
    
    +++
    id = "proj-a1b2"
    status = "open"
    deps = ["other-ticket"]
    links = []
    created = "2024-01-15T10:00:00Z"
    type = "task"
    priority = 2
    +++
    # Ticket Title
    
    Description here...

DEPENDENCIES:
    - Python 3.9+ (uses only stdlib)
    - opencode CLI (for run command)

EXAMPLES:
    # Run a single task with custom completion signal
    python basic_ralph.py run "Write tests" --completion-promise "TESTS_PASS"
    
    # Create a bug ticket with priority 0 (highest)
    python basic_ralph.py ticket create "Critical bug" -p 0 --type bug
    
    # Link related tickets
    python basic_ralph.py ticket link ticket1 ticket2 ticket3
    
    # Query tickets as JSON
    python basic_ralph.py ticket query
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# TOML support - tomllib is Python 3.11+, implement fallback for older versions
try:
    import tomllib

    HAS_TOMLLIB = True
except ImportError:
    HAS_TOMLLIB = False


TICKETS_DIR = ".tickets"


# =============================================================================
# TOML Utilities (stdlib has reader but no writer)
# =============================================================================


def dump_toml(data: dict) -> str:
    """Convert dict to TOML string (simple implementation for frontmatter)."""
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"[{key}]")
            for k, v in value.items():
                lines.append(_toml_value(k, v, 2))
        elif isinstance(value, list):
            if value:
                # Format arrays nicely
                if all(isinstance(x, str) for x in value):
                    lines.append(f"{key} = {json.dumps(value)}")
                else:
                    lines.append(f"{key} = {json.dumps(value)}")
            else:
                lines.append(f"{key} = []")
        else:
            lines.append(_toml_value(key, value, 0))
    return "\n".join(lines)


def _toml_value(key: str, value: Any, indent: int) -> str:
    """Format a single TOML key-value pair."""
    prefix = " " * indent
    if isinstance(value, str):
        return f"{prefix}{key} = {json.dumps(value)}"
    elif isinstance(value, bool):
        return f"{prefix}{key} = {str(value).lower()}"
    elif isinstance(value, (int, float)):
        return f"{prefix}{key} = {value}"
    elif isinstance(value, list):
        return f"{prefix}{key} = {json.dumps(value)}"
    else:
        return f"{prefix}{key} = {json.dumps(str(value))}"


def parse_toml(text: str) -> dict:
    """Simple TOML parser for basic frontmatter (fallback for tomllib)."""
    result = {}
    lines = text.strip().split("\n")
    current_table = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Table header
        if line.startswith("[") and line.endswith("]"):
            current_table = line[1:-1]
            result[current_table] = {}
            continue

        # Key-value pair
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Parse value
            parsed_value = _parse_toml_value(value)

            if current_table:
                result[current_table][key] = parsed_value
            else:
                result[key] = parsed_value

    return result


def _parse_toml_value(value_str: str) -> Any:
    """Parse a TOML value string."""
    value_str = value_str.strip()

    # String
    if (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        return json.loads(value_str)

    # Array
    if value_str.startswith("[") and value_str.endswith("]"):
        try:
            return json.loads(value_str)
        except:
            return []

    # Boolean
    if value_str == "true":
        return True
    if value_str == "false":
        return False

    # Integer
    try:
        return int(value_str)
    except ValueError:
        pass

    # Float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Date/Time (as string)
    return value_str


def load_toml(text: str) -> dict:
    """Load TOML, using tomllib if available, otherwise fallback."""
    if HAS_TOMLLIB:
        return tomllib.loads(text)
    else:
        return parse_toml(text)


# =============================================================================
# Ticket Helper Functions
# =============================================================================
def remove_ansi_escape_sequences(text):
    pattern = r"\x1b[^m]*m"
    return re.sub(pattern, "", text)


def ensure_dir():
    """Ensure tickets directory exists."""
    Path(TICKETS_DIR).mkdir(exist_ok=True)


def _iso_date():
    """Generate ISO date string."""
    return datetime.now(timezone.utc).isoformat()


def generate_id():
    """Generate ticket ID from directory name + timestamp hash."""
    dir_name = Path.cwd().name

    # Extract first letter of each hyphenated/underscored segment
    segments = re.split(r"[-_]", dir_name)
    prefix = "".join(segment[0].lower() if segment else "" for segment in segments)

    # Fallback to first 3 chars if no segments
    if not prefix:
        prefix = dir_name[:3].lower()

    # 4-char hash from timestamp + PID for entropy
    hash_input = f"{os.getpid()}{time.time()}".encode()
    hash_val = hashlib.sha256(hash_input).hexdigest()[:4]

    return f"{prefix}-{hash_val}"


def ticket_path(ticket_id: str):
    """Get ticket file path (supports partial ID matching)."""
    if not ticket_id:
        raise ValueError("Ticket ID cannot be empty")

    exact_path = Path(TICKETS_DIR) / f"{ticket_id}.md"

    if exact_path.exists():
        return exact_path

    # Try partial match (anywhere in filename)
    matches = list(Path(TICKETS_DIR).glob(f"*{ticket_id}*.md"))

    if len(matches) == 0:
        raise FileNotFoundError(f"Ticket '{ticket_id}' not found")
    elif len(matches) > 1:
        raise ValueError(f"ambiguous ID '{ticket_id}' matches multiple tickets")
    else:
        return matches[0]


def toml_field(file_path: Path, field: str):
    """Extract TOML field value from ticket file."""
    content = file_path.read_text()
    frontmatter_match = re.search(r"^\+\+\+(.*?)\+\+\+", content, re.DOTALL)

    if not frontmatter_match:
        return None

    try:
        frontmatter = load_toml(frontmatter_match.group(1))
        return frontmatter.get(field)
    except Exception:
        # Fallback: manual parsing
        frontmatter = frontmatter_match.group(1)
        lines = frontmatter.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith(f"{field} ="):
                value = line.split("=", 1)[1].strip()
                # Parse value
                if value.startswith('"') and value.endswith('"'):
                    return json.loads(value)
                elif value == "true":
                    return True
                elif value == "false":
                    return False
                elif value.startswith("[") and value.endswith("]"):
                    try:
                        return json.loads(value)
                    except:
                        return []
                else:
                    try:
                        return int(value)
                    except ValueError:
                        try:
                            return float(value)
                        except ValueError:
                            return value
    return None


def update_toml_field(file_path: Path, field: str, value):
    """Update TOML field in ticket file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Ticket file does not exist: {file_path}")

    content = file_path.read_text()

    # Check if field exists in frontmatter
    frontmatter_match = re.search(r"^(\+\+\+)(.*?)(\+\+\+)", content, re.DOTALL)

    if not frontmatter_match:
        raise ValueError("No TOML frontmatter found")

    frontmatter = frontmatter_match.group(2)
    frontmatter_lines = frontmatter.strip().split("\n")

    field_exists = False
    updated_lines = []

    for line in frontmatter_lines:
        stripped = line.strip()
        if stripped.startswith(f"{field} ="):
            updated_lines.append(_toml_value(field, value, 0))
            field_exists = True
        else:
            updated_lines.append(line)

    if not field_exists:
        # Add field to frontmatter
        updated_lines.append(_toml_value(field, value, 0))

    # Replace the frontmatter in the content
    updated_frontmatter = "\n".join(updated_lines)
    updated_content = content.replace(frontmatter_match.group(0), f"+++\n{updated_frontmatter}\n+++")

    file_path.write_text(updated_content)


def validate_status(status: str):
    """Validate if the status is one of the valid statuses."""
    valid_statuses = ["open", "in_progress", "closed"]
    if status not in valid_statuses:
        raise ValueError(f"invalid status '{status}'. Must be one of: {', '.join(valid_statuses)}")


def add_link_to_file(file_path: Path, target_id: str):
    """Add a link to another ticket in the links array."""
    current_links = toml_field(file_path, "links") or []

    # Skip if already present
    if target_id in current_links:
        return False

    current_links.append(target_id)
    update_toml_field(file_path, "links", current_links)
    return True


def remove_link_from_file(file_path: Path, target_id: str):
    """Remove a link to another ticket from the links array."""
    current_links = toml_field(file_path, "links") or []

    # Skip if not present
    if target_id not in current_links:
        return False

    current_links.remove(target_id)
    update_toml_field(file_path, "links", current_links)
    return True


def load_all_tickets():
    """Load all tickets from the tickets directory."""
    tickets = []
    tickets_dir = Path(TICKETS_DIR)

    if not tickets_dir.exists():
        return tickets

    for file_path in tickets_dir.glob("*.md"):
        try:
            content = file_path.read_text()

            # Extract frontmatter
            frontmatter_match = re.search(r"^\+\+\+(.*?)\+\+\+", content, re.DOTALL)
            if not frontmatter_match:
                continue

            try:
                frontmatter = load_toml(frontmatter_match.group(1))
            except Exception:
                continue

            # Extract title
            title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
            title = title_match.group(1) if title_match else "Untitled"

            ticket = {
                "id": frontmatter.get("id", file_path.stem),
                "status": frontmatter.get("status", "unknown"),
                "title": title,
                "deps": frontmatter.get("deps", []),
                "links": frontmatter.get("links", []),
                "priority": frontmatter.get("priority", 2),
                "file_path": file_path,
            }

            tickets.append(ticket)
        except Exception as e:
            print(
                f"Warning: Could not parse ticket file {file_path}: {str(e)}",
                file=sys.stderr,
            )

    return tickets


def print_tree_recursive(
    ticket_id,
    all_tickets,
    visited=None,
    depth=0,
    prefix="",
    is_last=True,
    full_mode=False,
):
    """Recursively print the dependency tree."""
    if visited is None:
        visited = set()

    if ticket_id in visited and not full_mode:
        return

    visited.add(ticket_id)

    # Get ticket info
    ticket = next((t for t in all_tickets if t["id"] == ticket_id), None)
    if not ticket:
        return

    # Print current ticket
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{ticket['id']} [{ticket['status']}] {ticket['title']}")

    # Prepare prefix for children
    new_prefix = prefix + ("    " if is_last else "│   ")

    # Print dependencies (children in the dependency tree)
    deps = ticket.get("deps", [])
    for i, dep_id in enumerate(deps):
        is_last_child = i == len(deps) - 1
        print_tree_recursive(
            dep_id,
            all_tickets,
            visited,
            depth + 1,
            new_prefix,
            is_last_child,
            full_mode,
        )


# =============================================================================
# Programmatic API for external use
# =============================================================================


def create_ticket(
    title: str = "Untitled",
    description: Optional[str] = None,
    design: Optional[str] = None,
    acceptance: Optional[str] = None,
    priority: int = 2,
    issue_type: str = "task",
    assignee: Optional[str] = None,
    external_ref: Optional[str] = None,
    parent: Optional[str] = None,
) -> str:
    """Create a new ticket programmatically and return the ticket ID.

    The ticket is created with 'complexity' field set to 'unknown'.
    This will be updated automatically when the task is executed with
    auto-classification enabled.
    """
    ensure_dir()

    # Validate priority range
    if priority < 0 or priority > 4:
        raise ValueError("Priority must be between 0 and 4")

    # Use git config user.name as default assignee if available
    if not assignee:
        try:
            result = subprocess.run(
                ["git", "config", "user.name"],
                capture_output=True,
                text=True,
                check=True,
            )
            assignee = result.stdout.strip()
        except subprocess.CalledProcessError:
            pass
        except FileNotFoundError:
            pass

    ticket_id = generate_id()
    file_path = Path(TICKETS_DIR) / f"{ticket_id}.md"
    now = _iso_date()

    # Build TOML frontmatter
    frontmatter = {
        "id": ticket_id,
        "status": "open",
        "deps": [],
        "links": [],
        "created": now,
        "type": issue_type,
        "priority": priority,
        "complexity": "unknown",
    }

    if assignee:
        frontmatter["assignee"] = assignee
    if external_ref:
        frontmatter["external-ref"] = external_ref
    if parent:
        # Verify parent exists if provided
        try:
            ticket_path(parent)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Parent ticket '{parent}' does not exist: {str(e)}")
        frontmatter["parent"] = parent

    # Build markdown content
    content = f"+++\n{dump_toml(frontmatter)}\n+++\n"
    content += f"# {title}\n\n"

    if description:
        content += f"{description}\n\n"
    if design:
        content += f"## Design\n\n{design}\n\n"
    if acceptance:
        content += f"## Acceptance Criteria\n\n{acceptance}\n\n"

    file_path.write_text(content)

    return ticket_id


def update_ticket_status(ticket_id: str, new_status: str):
    """Update ticket status programmatically."""
    validate_status(new_status)
    file_path = ticket_path(ticket_id)
    update_toml_field(file_path, "status", new_status)


def add_note_to_ticket(ticket_id: str, note: str):
    """Add a note to a ticket programmatically."""
    file_path = ticket_path(ticket_id)

    timestamp = _iso_date()

    # Read current content
    content = file_path.read_text()

    # Check if Notes section exists
    if not re.search(r"^## Notes\s*$", content, re.MULTILINE):
        # Add Notes section if missing
        content += "\n## Notes\n"

    # Append timestamped note
    content += f"\n**{timestamp}**\n\n{note}\n"

    # Write back to file
    file_path.write_text(content)


def get_ticket_info(ticket_id: str) -> dict:
    """Get ticket information programmatically.

    Returns a dict with ticket metadata including:
    - id, status, title, priority, type, complexity
    - assignee, deps, links, file_path
    """
    file_path = ticket_path(ticket_id)
    content = file_path.read_text()

    # Extract frontmatter
    frontmatter_match = re.search(r"^\+\+\+(.*?)\+\+\+", content, re.DOTALL)
    if not frontmatter_match:
        raise ValueError("No TOML frontmatter found in ticket")

    try:
        frontmatter = load_toml(frontmatter_match.group(1))
    except Exception as e:
        raise ValueError(f"Could not parse TOML frontmatter: {e}")

    # Extract title
    title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
    title = title_match.group(1) if title_match else "Untitled"

    return {
        "id": frontmatter.get("id", file_path.stem),
        "status": frontmatter.get("status", "unknown"),
        "title": title,
        "priority": frontmatter.get("priority", 2),
        "type": frontmatter.get("type", "task"),
        "complexity": frontmatter.get("complexity", "unknown"),
        "assignee": frontmatter.get("assignee"),
        "deps": frontmatter.get("deps", []),
        "links": frontmatter.get("links", []),
        "file_path": str(file_path),
    }


def list_all_tickets() -> List[dict]:
    """List all tickets programmatically."""
    return load_all_tickets()


# =============================================================================
# Basic Ralph - Research-Plan-Implement Functions
# =============================================================================


def run_opencode(prompt: str, model: Optional[str] = None) -> Tuple[str, int]:
    """Run opencode with the given prompt and return (stdout, exit_code)."""
    cmd = ["opencode", "run"]
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    if process.stdout:
        for line in process.stdout:
            print(line, end="")
            output_lines.append(line)

    process.wait()
    return "".join(output_lines), process.returncode


def ensure_artifacts_dir(ticket_id: str) -> Path:
    """Ensure artifacts directory exists for a ticket."""
    artifacts_dir = Path(TICKETS_DIR) / "artifacts" / ticket_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def construct_research_prompt(task_description: str, prior_research: Optional[str] = None) -> str:
    """Construct the research phase prompt."""
    prompt = f"""Research the following task and create a comprehensive research.md document:

Task: {task_description}

Your research should include:
1. Understanding the problem or requirements
2. Identifying relevant technologies, libraries, and approaches
3. Exploring existing code or patterns in the codebase
4. Identifying potential challenges or edge cases
5. Any additional context needed

Create a detailed research.md file with your findings.

Signal completion by outputting: READY_FOR_NEXT_TASK
"""
    if prior_research:
        prompt += f"\n\nPrior Research Context:\n{prior_research}"

    return prompt


def classify_task_complexity(task_description: str, model: Optional[str] = None) -> str:
    """Classify task complexity as 'simple', 'medium', or 'complex'.

    Analyzes the task description to determine appropriate workflow:
    - simple: Single file change, well-understood pattern, no new dependencies,
      clear requirements, < 10 lines of code. Skip research/planning.
    - medium: Multiple files, some research needed, potential edge cases,
      uses existing patterns. Use full workflow.
    - complex: New architecture, multiple components, significant research required,
      novel patterns or dependencies. Use full workflow with extra care.

    Args:
        task_description: The task description to classify
        model: Optional model override for the classification

    Returns:
        One of: 'simple', 'medium', 'complex'
    """
    classification_prompt = f"""Analyze this software development task and classify its complexity.

Task: {task_description}

Classify based on these criteria:
- **simple**: Single file change OR typo fix OR simple rename OR adding a log statement OR 
  well-understood pattern with clear requirements, minimal risk, < 20 lines of code change.
- **medium**: Multiple files affected, some research needed, potential edge cases to consider,
  uses existing patterns in the codebase, moderate risk.
- **complex**: New architecture required, multiple interconnected components, significant research needed,
  novel patterns or external dependencies, high risk, or unclear requirements.

Respond with ONLY ONE WORD from: simple, medium, or complex"""

    try:
        output, exit_code = run_opencode(classification_prompt, model)
        if exit_code != 0:
            print(f"Warning: Complexity classification failed (exit {exit_code}), defaulting to 'medium'")
            return "medium"

        # Extract the classification from output
        result = output.strip().lower()
        # Handle cases where model outputs more than just the word
        for complexity in ["simple", "medium", "complex"]:
            if complexity in result:
                return complexity

        # Default to medium if unclear
        print(f"Warning: Unclear complexity classification ('{result[:50]}...'), defaulting to 'medium'")
        return "medium"
    except Exception as e:
        print(f"Warning: Complexity classification error ({e}), defaulting to 'medium'")
        return "medium"


def construct_planning_prompt(task_description: str, research_content: str) -> str:
    """Construct the planning phase prompt."""
    return f"""Based on the following research, create a detailed implementation plan:

Task: {task_description}

Research Findings:
{research_content}

Create a comprehensive plan.md document that includes:
1. Implementation approach and strategy
2. Step-by-step breakdown of changes
3. Files to modify or create
4. Testing approach
5. Potential risks and mitigations

Signal completion by outputting: READY_FOR_NEXT_TASK
"""


def construct_implementation_prompt(
    task_description: str,
    research_content: str,
    plan_content: str,
    completion_promise: str = "COMPLETE",
) -> str:
    """Construct the implementation phase prompt."""
    return f"""Implement the following task based on the research and plan:

Task: {task_description}

Research Findings:
{research_content}

Implementation Plan:
{plan_content}

Execute the plan:
1. Make all necessary code changes
2. Ensure tests pass (if applicable)
3. Verify the implementation matches the requirements

Signal completion by outputting: {completion_promise}
"""


def check_phase_completion(output: str, success_signal: str) -> bool:
    """Check if the phase completed successfully based on output."""
    return success_signal in output


def save_artifact(ticket_id: str, filename: str, content: str):
    """Save an artifact file for a ticket."""
    artifacts_dir = ensure_artifacts_dir(ticket_id)
    artifact_path = artifacts_dir / filename
    artifact_path.write_text(remove_ansi_escape_sequences(content))
    return artifact_path


def load_artifact(ticket_id: str, filename: str) -> str:
    """Load an artifact file for a ticket."""
    artifact_path = Path(TICKETS_DIR) / "artifacts" / ticket_id / filename
    if artifact_path.exists():
        return artifact_path.read_text()
    return ""


def generate_task_summary(
    ticket_id: str,
    task_description: str,
    success: bool,
    implementation_output: str = "",
) -> str:
    """Generate a summary of the task execution and save to summary.md."""
    research_content = load_artifact(ticket_id, "research.md")
    plan_content = load_artifact(ticket_id, "plan.md")

    summary_lines = [
        f"# Task Summary: {ticket_id}",
        "",
        f"**Status:** {'Completed' if success else 'Incomplete'}",
        f"**Date:** {_iso_date()}",
        "",
        "## Task Description",
        "",
        task_description,
        "",
    ]

    if research_content:
        summary_lines.extend(
            [
                "## Research Summary",
                "",
                research_content[:1000] + ("..." if len(research_content) > 1000 else ""),
                "",
            ]
        )

    if plan_content:
        summary_lines.extend(
            [
                "## Implementation Plan",
                "",
                plan_content[:800] + ("..." if len(plan_content) > 800 else ""),
                "",
            ]
        )

    if implementation_output:
        summary_lines.extend(
            [
                "## Implementation Output",
                "",
                implementation_output[:2000] + ("..." if len(implementation_output) > 2000 else ""),
                "",
            ]
        )

    summary_content = "\n".join(summary_lines)

    save_artifact(ticket_id, "summary.md", summary_content)

    return summary_content


def extract_keywords(text: str) -> set:
    """Extract keywords from text for relevance matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    stopwords = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "just",
        "don",
        "now",
        "and",
        "or",
        "but",
        "if",
        "else",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "your",
        "yours",
        "me",
        "us",
        "him",
        "her",
        "his",
    }
    return {w for w in words if len(w) > 2 and w not in stopwords}


def is_related(ticket: dict, prompt: str) -> bool:
    """Check if a ticket is related to the prompt based on keyword matching."""
    ticket_title = ticket.get("title", "").lower()
    ticket_desc = ""
    if ticket.get("file_path"):
        try:
            content = Path(ticket["file_path"]).read_text()
            match = re.search(r"^# .+\n\n(.+?)(?:\n##|$)", content, re.DOTALL | re.MULTILINE)
            if match:
                ticket_desc = match.group(1).lower()
        except Exception:
            pass

    prompt_keywords = extract_keywords(prompt)
    title_keywords = extract_keywords(ticket_title)
    desc_keywords = extract_keywords(ticket_desc)

    if not prompt_keywords:
        return False

    common_with_title = prompt_keywords & title_keywords
    common_with_desc = prompt_keywords & desc_keywords

    if common_with_title and len(common_with_title) >= 1:
        return True
    if common_with_desc and len(common_with_desc) >= 2:
        return True

    return False


def find_related_incomplete_tasks(prompt: str) -> List[dict]:
    """Find incomplete (open/in_progress) tickets related to the given prompt."""
    all_tickets = load_all_tickets()
    related_tasks = []

    for ticket in all_tickets:
        status = ticket.get("status", "unknown")
        if status not in ["open", "in_progress"]:
            continue

        if is_related(ticket, prompt):
            related_tasks.append(ticket)

    related_tasks.sort(key=lambda t: (t.get("priority", 2), t["id"]))
    return related_tasks


def find_additional_tasks(completed_task_description: str) -> List[str]:
    """Find additional related tasks that should be done after completing a task."""
    all_tickets = load_all_tickets()
    additional_tasks = []

    for ticket in all_tickets:
        status = ticket.get("status", "unknown")
        if status not in ["open", "in_progress"]:
            continue

        if is_related(ticket, completed_task_description):
            additional_tasks.append(ticket["title"])

    return additional_tasks


def execute_phase(
    phase_name: str,
    prompt: str,
    ticket_id: str,
    artifact_filename: Optional[str] = None,
    model: Optional[str] = None,
    success_signal: str = "READY_FOR_NEXT_TASK",
    abort_signal: Optional[str] = None,
) -> Tuple[bool, str]:
    """Execute a phase and save artifacts if needed."""
    print(f"\n{'=' * 60}")
    print(f"Starting {phase_name} phase for ticket {ticket_id}")
    print(f"{'=' * 60}\n")

    output, exit_code = run_opencode(prompt, model)

    print(output)

    if exit_code != 0:
        print(f"\n⚠️  {phase_name} phase exited with code {exit_code}")
        return False, output

    if abort_signal and abort_signal in output:
        print(f"\n🛑 {phase_name} phase signaled abort: {abort_signal}")
        return False, output

    if artifact_filename:
        artifact_path = save_artifact(ticket_id, artifact_filename, output)
        print(f"\n📄 Saved artifact: {artifact_path}")

    if success_signal and success_signal not in output:
        print(f"\n⚠️  {phase_name} phase did not signal completion ({success_signal})")
        # Continue anyway if output was produced

    return True, output


def execute_task(
    task_description: str,
    ticket_id: str,
    model: Optional[str] = None,
    completion_promise: str = "COMPLETE",
    abort_promise: Optional[str] = None,
    task_promise: str = "READY_FOR_NEXT_TASK",
    fast: bool = False,
    auto_classify: bool = True,
) -> bool:
    """Execute the research-plan-implement workflow for a single task.

    Args:
        task_description: Description of the task to execute
        ticket_id: The ticket ID to track this task
        model: Optional model override for opencode
        completion_promise: Text signal that indicates task completion
        abort_promise: Text signal that indicates early abort
        task_promise: Text signal that indicates phase completion
        fast: If True, skip research/planning and go directly to implementation
        auto_classify: If True (default), automatically classify task complexity
                      and skip research/planning for simple tasks

    Returns:
        True if task completed successfully, False otherwise
    """
    # Auto-classify task complexity if enabled
    if auto_classify and not fast:
        print(f"\n{'=' * 60}")
        print(f"Classifying task complexity for ticket {ticket_id}...")
        print(f"{'=' * 60}")

        complexity = classify_task_complexity(task_description, model)

        # Store complexity in ticket metadata
        ticket_file = Path(TICKETS_DIR) / f"{ticket_id}.md"
        update_toml_field(ticket_file, "complexity", complexity)

        print(f"\n📊 Task classified as: {complexity.upper()}")

        if complexity == "simple":
            fast = True
            print("🚀 Fast mode enabled - skipping research and planning phases")
            add_note_to_ticket(
                ticket_id,
                f"Auto-classified as '{complexity}' - using fast mode (skipping research/planning)",
            )
        else:
            add_note_to_ticket(
                ticket_id,
                f"Auto-classified as '{complexity}' - using full workflow with research and planning",
            )

    if fast:
        print(f"\n{'=' * 60}")
        print(f"FAST MODE: Direct implementation for ticket {ticket_id}")
        print(f"{'=' * 60}\n")

        # Create minimal research.md in fast mode
        fast_research_content = f"""# Research: {task_description}

**Mode:** Fast Mode (auto-classified as simple or --fast flag used)
**Ticket:** {ticket_id}
**Date:** {_iso_date()}

## Summary

This task was executed in fast mode, skipping the full research phase.
The task was classified as simple enough to proceed directly to implementation.

## Task Description

{task_description}

## Notes

- Research phase skipped due to fast mode
- Task classified as simple/low complexity
- Proceeding directly to implementation
"""
        save_artifact(ticket_id, "research.md", fast_research_content)
        print(f"📄 Created research.md (fast mode)")

        # Create minimal plan.md in fast mode
        fast_plan_content = f"""# Implementation Plan: {task_description}

**Mode:** Fast Mode (auto-classified as simple or --fast flag used)
**Ticket:** {ticket_id}
**Date:** {_iso_date()}

## Summary

This task was executed in fast mode, skipping the full planning phase.
The task was classified as simple enough to proceed directly to implementation.

## Approach

Direct implementation based on task requirements.

## Task Description

{task_description}

## Notes

- Planning phase skipped due to fast mode
- Task classified as simple/low complexity
- Proceeding directly to implementation
"""
        save_artifact(ticket_id, "plan.md", fast_plan_content)
        print(f"📄 Created plan.md (fast mode)")

        implementation_prompt = f"""Execute the following task:

{task_description}

Signal completion by outputting: {completion_promise}
"""
        success, implementation_output = execute_phase(
            "Implementation",
            implementation_prompt,
            ticket_id,
            model=model,
            success_signal=completion_promise,
            abort_signal=abort_promise,
        )

        if success and completion_promise in implementation_output:
            update_ticket_status(ticket_id, "closed")
            generate_task_summary(ticket_id, task_description, True, implementation_output)
            add_note_to_ticket(
                ticket_id,
                f"Task completed successfully (fast mode).\n\nArtifacts saved to .tickets/artifacts/{ticket_id}/",
            )
            print(f"\n✅ Task completed and ticket {ticket_id} closed")
        else:
            generate_task_summary(ticket_id, task_description, False, implementation_output)
            add_note_to_ticket(
                ticket_id,
                f"Task execution finished (fast mode).\n\nOutput:\n{implementation_output[:2000]}",
            )
            print(f"\n⚠️  Task execution finished but completion signal not detected")

        return success

    # Phase 1: Research
    research_prompt = construct_research_prompt(task_description)
    success, research_output = execute_phase(
        "Research",
        research_prompt,
        ticket_id,
        artifact_filename="research.md",
        model=model,
        success_signal=task_promise,
        abort_signal=abort_promise,
    )

    if not success:
        add_note_to_ticket(
            ticket_id,
            f"Research phase failed or was aborted.\n\nOutput:\n{research_output[:1000]}",
        )
        return False

    # Load research content
    research_content = load_artifact(ticket_id, "research.md")

    # Phase 2: Planning
    planning_prompt = construct_planning_prompt(task_description, research_content)
    success, planning_output = execute_phase(
        "Planning",
        planning_prompt,
        ticket_id,
        artifact_filename="plan.md",
        model=model,
        success_signal=task_promise,
        abort_signal=abort_promise,
    )

    if not success:
        add_note_to_ticket(
            ticket_id,
            f"Planning phase failed or was aborted.\n\nOutput:\n{planning_output[:1000]}",
        )
        return False

    # Load plan content
    plan_content = load_artifact(ticket_id, "plan.md")

    # Phase 3: Implementation
    implementation_prompt = construct_implementation_prompt(
        task_description,
        research_content,
        plan_content,
        completion_promise,
    )
    success, implementation_output = execute_phase(
        "Implementation",
        implementation_prompt,
        ticket_id,
        model=model,
        success_signal=completion_promise,
        abort_signal=abort_promise,
    )

    # Update ticket with completion status
    if success and completion_promise in implementation_output:
        update_ticket_status(ticket_id, "closed")
        generate_task_summary(ticket_id, task_description, True, implementation_output)
        add_note_to_ticket(
            ticket_id,
            f"Task completed successfully.\n\nArtifacts saved to .tickets/artifacts/{ticket_id}/\n- research.md\n- plan.md\n- summary.md",
        )
        print(f"\n✅ Task completed and ticket {ticket_id} closed")
    else:
        generate_task_summary(ticket_id, task_description, False, implementation_output)
        add_note_to_ticket(
            ticket_id,
            f"Task execution finished.\n\nOutput:\n{implementation_output[:2000]}",
        )
        print(f"\n⚠️  Task execution finished but completion signal not detected")

    return success


# =============================================================================
# CLI Command Handlers - Ticket Commands
# =============================================================================


def cmd_ticket_create(args):
    """Create a new ticket."""
    try:
        ticket_id = create_ticket(
            title=args.title,
            description=args.description,
            design=args.design,
            acceptance=args.acceptance,
            priority=args.priority,
            issue_type=args.type,
            assignee=args.assignee,
            external_ref=args.external_ref,
            parent=args.parent,
        )
        print(ticket_id)
    except Exception as e:
        print(f"Error creating ticket: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ticket_status(args):
    """Update ticket status."""
    try:
        validate_status(args.new_status)
        file_path = ticket_path(args.ticket_id)
        update_toml_field(file_path, "status", args.new_status)
        print(f"Updated {file_path.stem} -> {args.new_status}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ticket_start(args):
    """Set ticket status to in_progress."""
    try:
        file_path = ticket_path(args.ticket_id)
        update_toml_field(file_path, "status", "in_progress")
        print(f"Updated {file_path.stem} -> in_progress")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ticket_close(args):
    """Set ticket status to closed."""
    try:
        file_path = ticket_path(args.ticket_id)
        update_toml_field(file_path, "status", "closed")
        print(f"Updated {file_path.stem} -> closed")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ticket_reopen(args):
    """Set ticket status to open."""
    try:
        file_path = ticket_path(args.ticket_id)
        update_toml_field(file_path, "status", "open")
        print(f"Updated {file_path.stem} -> open")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ticket_dep(args):
    """Add dependency (ticket_id depends on dep_id)."""
    try:
        file_path = ticket_path(args.ticket_id)

        # Verify dependency exists
        ticket_path(args.dep_id)

        # Get current deps
        current_deps = toml_field(file_path, "deps") or []

        # Add dep if not already present
        if args.dep_id in current_deps:
            print("Dependency already exists")
            return

        # Update deps array
        current_deps.append(args.dep_id)
        update_toml_field(file_path, "deps", current_deps)

        print(f"Added dependency: {file_path.stem} -> {args.dep_id}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ticket_undep(args):
    """Remove dependency."""
    try:
        file_path = ticket_path(args.ticket_id)

        # Get current deps
        current_deps = toml_field(file_path, "deps") or []

        # Check if dependency exists
        if args.dep_id not in current_deps:
            print("Dependency not found", file=sys.stderr)
            sys.exit(1)

        # Remove dep from array
        current_deps.remove(args.dep_id)
        update_toml_field(file_path, "deps", current_deps)

        print(f"Removed dependency: {file_path.stem} -/-> {args.dep_id}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ticket_link(args):
    """Link tickets together (symmetric)."""
    if len(args.ticket_ids) < 2:
        print("Usage: ticket link <id> <id> [id...]", file=sys.stderr)
        sys.exit(1)

    try:
        # Resolve all ticket paths first
        files = []
        ids = []
        for ticket_id in args.ticket_ids:
            file_path = ticket_path(ticket_id)
            files.append(file_path)
            ids.append(file_path.stem)

        # Update each file to include links to all other tickets
        added_count = 0
        for i, file_path in enumerate(files):
            # Build list of other IDs to link
            others = [id for j, id in enumerate(ids) if i != j]

            # Add missing links
            for other_id in others:
                if add_link_to_file(file_path, other_id):
                    added_count += 1

        if added_count == 0:
            print("All links already exist")
        else:
            print(f"Added {added_count} link(s) between {len(ids)} tickets")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ticket_unlink(args):
    """Remove link between tickets."""
    try:
        file_path = ticket_path(args.ticket_id)
        target_file_path = ticket_path(args.target_id)

        # Remove from both files
        removed_from_first = remove_link_from_file(file_path, args.target_id)
        removed_from_second = remove_link_from_file(target_file_path, args.ticket_id)

        if not removed_from_first and not removed_from_second:
            print(
                f"No link found between {file_path.stem} and {args.target_id}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Removed link: {file_path.stem} <-> {args.target_id}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ticket_ls(args):
    """List tickets."""
    tickets = load_all_tickets()

    for ticket in tickets:
        if args.status and ticket["status"] != args.status:
            continue

        deps_str = f" <- [{', '.join(ticket['deps'])}]" if ticket["deps"] else ""
        print(f"{ticket['id']:<8} [{ticket['status']}] - {ticket['title']}{deps_str}")


def cmd_ticket_ready(args):
    """List open/in-progress tickets with deps resolved."""
    tickets = load_all_tickets()

    # Create a mapping of ticket id to status for dependency checking
    status_map = {ticket["id"]: ticket["status"] for ticket in tickets}

    ready_tickets = []
    for ticket in tickets:
        status = ticket["status"]
        if status not in ["open", "in_progress"]:
            continue

        # Check if all dependencies are closed
        deps = ticket.get("deps", [])
        all_deps_closed = True
        for dep_id in deps:
            dep_status = status_map.get(dep_id)
            if dep_status != "closed":
                all_deps_closed = False
                break

        if all_deps_closed:
            ready_tickets.append(ticket)

    # Sort by priority, then by id
    ready_tickets.sort(key=lambda t: (t["priority"], t["id"]))

    for ticket in ready_tickets:
        print(f"{ticket['id']:<8} [P{ticket['priority']}][{ticket['status']}] - {ticket['title']}")


def cmd_ticket_blocked(args):
    """List open/in-progress tickets with unresolved deps."""
    tickets = load_all_tickets()

    # Create a mapping of ticket id to status for dependency checking
    status_map = {ticket["id"]: ticket["status"] for ticket in tickets}

    blocked_tickets = []
    for ticket in tickets:
        status = ticket["status"]
        if status not in ["open", "in_progress"]:
            continue

        # Check if any dependencies are not closed
        deps = ticket.get("deps", [])
        open_deps = []
        for dep_id in deps:
            dep_status = status_map.get(dep_id)
            if dep_status != "closed":
                open_deps.append(dep_id)

        if open_deps:
            ticket_copy = ticket.copy()
            ticket_copy["open_deps"] = open_deps
            blocked_tickets.append(ticket_copy)

    # Sort by priority, then by id
    blocked_tickets.sort(key=lambda t: (t["priority"], t["id"]))

    for ticket in blocked_tickets:
        blockers_str = f" <- [{', '.join(ticket['open_deps'])}]"
        print(f"{ticket['id']:<8} [P{ticket['priority']}][{ticket['status']}] - {ticket['title']}{blockers_str}")


def cmd_ticket_closed(args):
    """List recently closed tickets."""
    tickets_dir = Path(TICKETS_DIR)

    if not tickets_dir.exists():
        return

    # Get all ticket files sorted by modification time (most recent first)
    ticket_files = sorted(tickets_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)

    closed_tickets = []
    for file_path in ticket_files[:100]:  # Check max 100 most recent files
        content = file_path.read_text()

        # Extract frontmatter
        frontmatter_match = re.search(r"^\+\+\+(.*?)\+\+\+", content, re.DOTALL)
        if not frontmatter_match:
            continue

        try:
            frontmatter = load_toml(frontmatter_match.group(1))
        except Exception:
            continue

        # Extract title
        title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Untitled"

        status = frontmatter.get("status", "unknown")
        if status in ["closed", "done"]:
            ticket = {
                "id": frontmatter.get("id", file_path.stem),
                "status": status,
                "title": title,
            }
            closed_tickets.append(ticket)

    # Limit results
    for ticket in closed_tickets[: args.limit]:
        print(f"{ticket['id']:<8} [{ticket['status']}] - {ticket['title']}")


def cmd_ticket_show(args):
    """Display ticket details."""
    try:
        file_path = ticket_path(args.ticket_id)
        target_id = file_path.stem

        # Load all tickets to find relationships
        all_tickets = load_all_tickets()
        ticket_map = {ticket["id"]: ticket for ticket in all_tickets}

        # Read the target ticket file
        content = file_path.read_text()

        # Find and display the main content (excluding frontmatter)
        content_lines = content.split("\n")
        frontmatter_end = -1
        for i, line in enumerate(content_lines):
            if line.strip() == "+++" and i > 0:  # This is the end of frontmatter
                frontmatter_end = i
                break

        # Print content from after the frontmatter
        for line in content_lines[frontmatter_end + 1 :]:
            print(line)

        # Find blockers (unclosed deps)
        target_ticket = ticket_map.get(target_id)
        if target_ticket:
            blockers = []
            for dep_id in target_ticket.get("deps", []):
                dep_ticket = ticket_map.get(dep_id)
                if dep_ticket and dep_ticket["status"] != "closed":
                    blockers.append(dep_ticket)

            if blockers:
                print("\n## Blockers\n")
                for blocker in blockers:
                    print(f"- {blocker['id']} [{blocker['status']}] {blocker['title']}")

            # Find tickets this is blocking
            blocking = []
            for ticket in all_tickets:
                if target_id in ticket.get("deps", []):
                    if ticket["status"] != "closed":
                        blocking.append(ticket)

            if blocking:
                print("\n## Blocking\n")
                for blocked in blocking:
                    print(f"- {blocked['id']} [{blocked['status']}] {blocked['title']}")

            # Find children (tickets with this as parent)
            children = []
            for ticket in all_tickets:
                parent_id = toml_field(ticket["file_path"], "parent")
                if parent_id == target_id:
                    children.append(ticket)

            if children:
                print("\n## Children\n")
                for child in children:
                    print(f"- {child['id']} [{child['status']}] {child['title']}")

            # Find linked tickets
            linked = []
            for link_id in target_ticket.get("links", []):
                link_ticket = ticket_map.get(link_id)
                if link_ticket:
                    linked.append(link_ticket)

            if linked:
                print("\n## Linked\n")
                for link in linked:
                    print(f"- {link['id']} [{link['status']}] {link['title']}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ticket_edit(args):
    """Open ticket in editor."""
    try:
        file_path = ticket_path(args.ticket_id)

        editor = os.environ.get("EDITOR", "vi")
        subprocess.run([editor, str(file_path)])
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ticket_add_note(args):
    """Append timestamped note to ticket."""
    try:
        file_path = ticket_path(args.ticket_id)

        if args.note is None:
            # Read from stdin if no note provided
            if sys.stdin.isatty():  # If stdin is a terminal
                print("Error: no note provided", file=sys.stderr)
                sys.exit(1)
            else:
                note = sys.stdin.read()
        else:
            note = args.note

        timestamp = _iso_date()

        # Read current content
        content = file_path.read_text()

        # Check if Notes section exists
        if not re.search(r"^## Notes\s*$", content, re.MULTILINE):
            # Add Notes section if missing
            content += "\n## Notes\n"

        # Append timestamped note
        content += f"\n**{timestamp}**\n\n{note}\n"

        # Write back to file
        file_path.write_text(content)

        print(f"Note added to {file_path.stem}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ticket_query(args):
    """Output tickets as JSON."""
    tickets_dir = Path(TICKETS_DIR)

    if not tickets_dir.exists():
        return

    # Collect all tickets
    all_tickets = []
    for file_path in tickets_dir.glob("*.md"):
        content = file_path.read_text()

        # Extract frontmatter
        frontmatter_match = re.search(r"^\+\+\+(.*?)\+\+\+", content, re.DOTALL)
        if not frontmatter_match:
            continue

        try:
            frontmatter = load_toml(frontmatter_match.group(1))
        except Exception:
            continue

        all_tickets.append(frontmatter)

    # Output all tickets as JSON
    for ticket in all_tickets:
        print(json.dumps(ticket))


def cmd_ticket_dep_tree(args):
    """Show dependency tree."""
    try:
        # Load all tickets
        all_tickets = load_all_tickets()

        # Find the root ticket
        root_ticket = next((t for t in all_tickets if args.ticket_id in t["id"]), None)
        if not root_ticket:
            # Try partial match
            matches = [t for t in all_tickets if args.ticket_id in t["id"]]
            if len(matches) == 0:
                print(f"Error: ticket '{args.ticket_id}' not found", file=sys.stderr)
                sys.exit(1)
            elif len(matches) > 1:
                print(
                    f"Error: ambiguous ID '{args.ticket_id}' matches multiple tickets",
                    file=sys.stderr,
                )
                sys.exit(1)
            root_ticket = matches[0]
            ticket_id = root_ticket["id"]
        else:
            ticket_id = root_ticket["id"]

        # Print root
        print(f"{ticket_id} [{root_ticket['status']}] {root_ticket['title']}")

        # Print dependencies recursively
        deps = root_ticket.get("deps", [])
        for i, dep_id in enumerate(deps):
            is_last = i == len(deps) - 1
            new_prefix = "    "
            print_tree_recursive(dep_id, all_tickets, set(), 1, new_prefix, is_last, args.full)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# CLI Command Handlers - Run Command
# =============================================================================


def cmd_run(args):
    """Execute the research-plan-implement workflow.

    By default, automatically classifies task complexity and skips
    research/planning for simple tasks (smart-fast mode). Use --full
    to force the complete workflow or --fast to force fast mode.
    """

    # Load all tasks
    task_descriptions = []

    if args.prompt_files:
        for file_path in args.prompt_files:
            path = Path(file_path)
            if path.exists():
                task_descriptions.append(path.read_text())
            else:
                print(f"Warning: Prompt file not found: {file_path}")

    if args.prompt:
        task_descriptions.append(args.prompt)

    if not task_descriptions:
        # Read from stdin if available
        if not sys.stdin.isatty():
            task_descriptions.append(sys.stdin.read())

    if not task_descriptions:
        print("Error: No tasks provided. Use --prompt-file or provide a prompt.")
        sys.exit(1)

    print(f"Basic Ralph starting with {len(task_descriptions)} task(s)")
    print(f"  Min iterations: {args.min_iterations}")
    print(f"  Max iterations: {args.max_iterations or 'unlimited'}")
    print(f"  Completion promise: {args.completion_promise}")
    print(f"  Abort promise: {args.abort_promise or 'none'}")
    print(f"  Tasks mode: {args.tasks}")
    print()

    def get_model_for_iteration(rotation: Optional[str], iteration: int, default_model: Optional[str]) -> Optional[str]:
        """Get the model to use for a specific iteration based on rotation."""
        if not rotation:
            return default_model

        models = [m.strip() for m in rotation.split(",")]
        if not models:
            return default_model

        return models[iteration % len(models)]

    def execute_single_task(
        task_desc: str,
        task_num: Union[int, str],
        total_tasks: int,
        placeholder_ticket_id: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """Execute a single task and return (ticket_id, completed)."""
        print(f"\n{'#' * 60}")
        print(f"# Task {task_num}/{total_tasks}")
        print(f"{'#' * 60}")

        ticket_title = task_desc.split("\n")[0][:50] + "..." if len(task_desc) > 50 else task_desc.split("\n")[0]

        if placeholder_ticket_id:
            ticket_id = placeholder_ticket_id
            update_ticket_status(ticket_id, "in_progress")
            update_toml_field(
                Path(TICKETS_DIR) / f"{ticket_id}.md",
                "title",
                f"Task {task_num}: {ticket_title}",
            )
        else:
            ticket_id = create_ticket(
                title=f"Task {task_num}: {ticket_title}",
                description=task_desc[:500],
                issue_type="task",
                priority=2,
            )
            update_ticket_status(ticket_id, "in_progress")

        print(f"Created ticket: {ticket_id}")

        max_iters = args.max_iterations
        min_iters = args.min_iterations

        iteration = 0
        completed = False

        while True:
            iteration += 1

            if max_iters and iteration > max_iters:
                print(f"\nReached max iterations ({max_iters}) for task {task_num}")
                add_note_to_ticket(ticket_id, f"Stopped after reaching max iterations ({max_iters})")
                break

            print(f"\n{'=' * 60}")
            print(f"Iteration {iteration} for task {task_num}")
            print(f"{'=' * 60}")

            model = get_model_for_iteration(args.rotation, iteration - 1, args.model)
            if model:
                print(f"Using model: {model}")

            success = execute_task(
                task_description=task_desc,
                ticket_id=ticket_id,
                model=model,
                completion_promise=args.completion_promise,
                abort_promise=args.abort_promise,
                task_promise=args.task_promise,
                fast=args.fast,
                auto_classify=not args.full,
            )

            ticket_info = get_ticket_info(ticket_id)
            if ticket_info["status"] == "closed":
                completed = True
                break

            if iteration < min_iters:
                print(f"Continuing (min iterations not yet reached: {iteration}/{min_iters})")
                continue

            if not success:
                print(f"\nTask did not complete successfully in iteration {iteration}")
                continue

            additional_tasks = find_additional_tasks(task_desc)
            if additional_tasks and max_iters and iteration < max_iters:
                print(f"\n⚡ Found {len(additional_tasks)} additional related task(s) to complete:")
                for i, add_task in enumerate(additional_tasks, 1):
                    print(f"  {i}. {add_task}")

                additional_ticket_id = create_ticket(
                    title=f"Additional task: {additional_tasks[0][:50]}",
                    description=f"Related to: {task_desc[:200]}",
                    issue_type="task",
                    priority=2,
                )
                update_ticket_status(additional_ticket_id, "in_progress")

                print(f"\n🔄 Processing additional related task: {additional_ticket_id}")

                additional_success = execute_single_task(
                    additional_tasks[0],
                    f"{task_num}.{iteration}",
                    total_tasks,
                    additional_ticket_id,
                )

                if not additional_success[1]:
                    print(f"\n⚠️  Additional task did not complete, continuing with current task")

        if completed:
            print(f"\n✅ Task {task_num} completed successfully (ticket: {ticket_id})")
        else:
            print(f"\n⚠️  Task {task_num} did not complete (ticket: {ticket_id} remains open)")

        return ticket_id, completed

    for task_idx, task_description in enumerate(task_descriptions, 1):
        related_tasks = find_related_incomplete_tasks(task_description)

        if related_tasks:
            print(f"\n🔍 Found {len(related_tasks)} related incomplete task(s) to complete first:")
            for i, rel_task in enumerate(related_tasks, 1):
                print(f"  {i}. [{rel_task['status']}] {rel_task['title']} ({rel_task['id']})")
            print()

            current_task_ticket_id = create_ticket(
                title=f"Task {task_idx}: {task_description.split(chr(10))[0][:50]}",
                description=f"Main task - related to: {', '.join(t['id'] for t in related_tasks)}",
                issue_type="task",
                priority=2,
            )

            for rel_idx, rel_task in enumerate(related_tasks, 1):
                print(f"\n{'=' * 60}")
                print(f"# Processing related incomplete task {rel_idx}/{len(related_tasks)}: {rel_task['id']}")
                print(f"{'=' * 60}")

                execute_single_task(rel_task["title"], f"R{rel_idx}", len(related_tasks), rel_task["id"])

            print(f"\n{'=' * 60}")
            print(f"# Now processing main task: {task_idx}")
            print(f"{'=' * 60}")

            execute_single_task(
                task_description,
                task_idx,
                len(task_descriptions),
                current_task_ticket_id,
            )
        else:
            execute_single_task(task_description, task_idx, len(task_descriptions))

    print(f"\n{'=' * 60}")
    print("Basic Ralph finished processing all tasks")
    print(f"{'=' * 60}")


# =============================================================================
# Main CLI Setup
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Basic Ralph - Research-Plan-Implement loop with ticket management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the research-plan-implement workflow:
  %(prog)s run "Implement feature X"
  
  # Create a new ticket:
  %(prog)s ticket create "Fix bug" -d "Description"
  
  # List all tickets:
  %(prog)s ticket ls
  
  # Show ticket details:
  %(prog)s ticket show abc-1234
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -------------------------------------------------------------------------
    # Run command (original basic_ralph workflow)
    # -------------------------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Execute research-plan-implement workflow (auto-classifies and skips phases for simple tasks by default)",
    )

    run_parser.add_argument(
        "--min-iterations",
        type=int,
        default=1,
        help="Minimum iterations before completion allowed (default: 1)",
    )

    run_parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Stop after N iterations (default: unlimited)",
    )

    run_parser.add_argument(
        "--completion-promise",
        type=str,
        default="COMPLETE",
        help="Text that signals completion (default: COMPLETE)",
    )

    run_parser.add_argument(
        "--abort-promise",
        type=str,
        default=None,
        help="Phrase that signals early abort (e.g., ABORT_FAILURE)",
    )

    run_parser.add_argument(
        "--tasks",
        "-t",
        action="store_true",
        help="Enable Tasks Mode for structured task tracking",
    )

    run_parser.add_argument(
        "--task-promise",
        type=str,
        default="READY_FOR_NEXT_TASK",
        help="Text that signals task completion (default: READY_FOR_NEXT_TASK)",
    )

    run_parser.add_argument("--model", type=str, default=None, help="Model to use (agent-specific)")

    run_parser.add_argument(
        "--rotation",
        type=str,
        default=None,
        help="Agent/model rotation for each iteration (comma-separated)",
    )

    run_parser.add_argument(
        "--prompt-file",
        "--file",
        "-f",
        action="append",
        dest="prompt_files",
        help="Read prompt content from a file (can be specified multiple times)",
    )

    run_parser.add_argument("--no-commit", action="store_true", help="Don't auto-commit after iterations")

    run_parser.add_argument(
        "--fast",
        action="store_true",
        help="Force skip research and planning, go directly to implementation",
    )

    run_parser.add_argument(
        "--full",
        action="store_true",
        help="Force full workflow with research and planning (disable auto-classify)",
    )

    run_parser.add_argument("prompt", nargs="?", help="Task description (if not using --prompt-file)")

    # -------------------------------------------------------------------------
    # Ticket command with nested subcommands
    # -------------------------------------------------------------------------
    ticket_parser = subparsers.add_parser("ticket", help="Ticket management commands")
    ticket_subparsers = ticket_parser.add_subparsers(dest="ticket_command", help="Ticket commands")

    # ticket create
    ticket_create = ticket_subparsers.add_parser("create", help="Create a new ticket")
    ticket_create.add_argument("title", nargs="?", default="Untitled", help="Title of the ticket")
    ticket_create.add_argument("-d", "--description", help="Description text")
    ticket_create.add_argument("--design", help="Design notes")
    ticket_create.add_argument("--acceptance", help="Acceptance criteria")
    ticket_create.add_argument("-p", "--priority", type=int, default=2, help="Priority 0-4, 0=highest")
    ticket_create.add_argument("-t", "--type", default="task", help="Type (bug|feature|task|epic|chore)")
    ticket_create.add_argument("-a", "--assignee", help="Assignee")
    ticket_create.add_argument("--external-ref", help="External reference (e.g., gh-123, JIRA-456)")
    ticket_create.add_argument("--parent", help="Parent ticket ID")

    # ticket status
    ticket_status = ticket_subparsers.add_parser("status", help="Update ticket status")
    ticket_status.add_argument("ticket_id", help="Ticket ID")
    ticket_status.add_argument("new_status", help="New status (open|in_progress|closed)")

    # ticket start
    ticket_start = ticket_subparsers.add_parser("start", help="Set ticket status to in_progress")
    ticket_start.add_argument("ticket_id", help="Ticket ID")

    # ticket close
    ticket_close = ticket_subparsers.add_parser("close", help="Set ticket status to closed")
    ticket_close.add_argument("ticket_id", help="Ticket ID")

    # ticket reopen
    ticket_reopen = ticket_subparsers.add_parser("reopen", help="Set ticket status to open")
    ticket_reopen.add_argument("ticket_id", help="Ticket ID")

    # ticket dep
    ticket_dep = ticket_subparsers.add_parser("dep", help="Add dependency (ticket_id depends on dep_id)")
    ticket_dep.add_argument("ticket_id", help="Ticket ID")
    ticket_dep.add_argument("dep_id", help="Dependency ticket ID")

    # ticket undep
    ticket_undep = ticket_subparsers.add_parser("undep", help="Remove dependency")
    ticket_undep.add_argument("ticket_id", help="Ticket ID")
    ticket_undep.add_argument("dep_id", help="Dependency ticket ID")

    # ticket link
    ticket_link = ticket_subparsers.add_parser("link", help="Link tickets together (symmetric)")
    ticket_link.add_argument("ticket_ids", nargs="+", help="Ticket IDs to link")

    # ticket unlink
    ticket_unlink = ticket_subparsers.add_parser("unlink", help="Remove link between tickets")
    ticket_unlink.add_argument("ticket_id", help="Ticket ID")
    ticket_unlink.add_argument("target_id", help="Target ticket ID")

    # ticket ls
    ticket_ls = ticket_subparsers.add_parser("ls", help="List tickets")
    ticket_ls.add_argument("--status", help="Filter by status")

    # ticket ready
    ticket_ready = ticket_subparsers.add_parser("ready", help="List open/in-progress tickets with deps resolved")

    # ticket blocked
    ticket_blocked = ticket_subparsers.add_parser("blocked", help="List open/in-progress tickets with unresolved deps")

    # ticket closed
    ticket_closed = ticket_subparsers.add_parser("closed", help="List recently closed tickets")
    ticket_closed.add_argument("--limit", type=int, default=20, help="Limit number of results")

    # ticket show
    ticket_show = ticket_subparsers.add_parser("show", help="Display ticket details")
    ticket_show.add_argument("ticket_id", help="Ticket ID")

    # ticket edit
    ticket_edit = ticket_subparsers.add_parser("edit", help="Open ticket in editor")
    ticket_edit.add_argument("ticket_id", help="Ticket ID")

    # ticket add-note
    ticket_add_note = ticket_subparsers.add_parser("add-note", help="Append timestamped note to ticket")
    ticket_add_note.add_argument("ticket_id", help="Ticket ID")
    ticket_add_note.add_argument("note", nargs="?", help="Note text (reads from stdin if not provided)")

    # ticket query
    ticket_query = ticket_subparsers.add_parser("query", help="Output tickets as JSON")
    ticket_query.add_argument("filter_expr", nargs="?", help=argparse.SUPPRESS)

    # ticket dep-tree
    ticket_dep_tree = ticket_subparsers.add_parser("dep-tree", help="Show dependency tree")
    ticket_dep_tree.add_argument("ticket_id", help="Ticket ID")
    ticket_dep_tree.add_argument("--full", action="store_true", help="Show full tree without deduplication")

    # Parse args
    args = parser.parse_args()

    # Route to appropriate handler
    if args.command == "run":
        cmd_run(args)
    elif args.command == "ticket":
        if args.ticket_command == "create":
            cmd_ticket_create(args)
        elif args.ticket_command == "status":
            cmd_ticket_status(args)
        elif args.ticket_command == "start":
            cmd_ticket_start(args)
        elif args.ticket_command == "close":
            cmd_ticket_close(args)
        elif args.ticket_command == "reopen":
            cmd_ticket_reopen(args)
        elif args.ticket_command == "dep":
            cmd_ticket_dep(args)
        elif args.ticket_command == "undep":
            cmd_ticket_undep(args)
        elif args.ticket_command == "link":
            cmd_ticket_link(args)
        elif args.ticket_command == "unlink":
            cmd_ticket_unlink(args)
        elif args.ticket_command == "ls":
            cmd_ticket_ls(args)
        elif args.ticket_command == "ready":
            cmd_ticket_ready(args)
        elif args.ticket_command == "blocked":
            cmd_ticket_blocked(args)
        elif args.ticket_command == "closed":
            cmd_ticket_closed(args)
        elif args.ticket_command == "show":
            cmd_ticket_show(args)
        elif args.ticket_command == "edit":
            cmd_ticket_edit(args)
        elif args.ticket_command == "add-note":
            cmd_ticket_add_note(args)
        elif args.ticket_command == "query":
            cmd_ticket_query(args)
        elif args.ticket_command == "dep-tree":
            cmd_ticket_dep_tree(args)
        else:
            ticket_parser.print_help()
    else:
        # Default to run command for backward compatibility if prompt provided
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            # Looks like a prompt, run the workflow
            args = run_parser.parse_args(sys.argv[1:])
            cmd_run(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
