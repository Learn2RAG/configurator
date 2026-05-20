"""
jira_loader.py

Description:
This module handles loading documents from Jira issues via the Jira REST API.
Supports authentication (none, Basic Auth, Bearer token), pagination,
and incremental loading based on the issue update timestamp.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.1
Creation Date: May 18, 2026
Last Modified: May 18, 2026
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Union

import requests
from langchain_core.documents import Document

from ..globals import stop_loading

logger = logging.getLogger("Learn2RAGImporter")


def _build_session(auth_type: str, username: str, password: str, token: str) -> requests.Session:
    """Build a requests session for Jira API access."""
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    if auth_type == "basic":
        session.auth = (username, password)
    elif auth_type == "token":
        session.headers.update({"Authorization": f"Bearer {token}"})

    return session


def _build_base_jql(
    jql: str,
    projects: Optional[List[str]] = None,
    issue_types: Optional[List[str]] = None,
) -> str:
    """Build the base JQL from explicit jql or from project/issue type filters."""
    if jql.strip():
        return jql.strip()

    clauses: List[str] = []

    if projects:
        quoted_projects = ",".join(f'"{project}"' for project in projects if str(project).strip())
        if quoted_projects:
            clauses.append(f"project in ({quoted_projects})")

    if issue_types:
        quoted_types = ",".join(f'"{issue_type}"' for issue_type in issue_types if str(issue_type).strip())
        if quoted_types:
            clauses.append(f"issuetype in ({quoted_types})")

    if not clauses:
        return "ORDER BY updated DESC"

    return " AND ".join(clauses) + " ORDER BY updated DESC"


def _add_since_to_jql(base_jql: str, since: Optional[datetime]) -> str:
    """Append an updated timestamp filter to a JQL query."""
    if since is None:
        return base_jql

    since_utc = since.astimezone(timezone.utc) if since.tzinfo else since.replace(tzinfo=timezone.utc)
    since_str = since_utc.strftime("%Y-%m-%d %H:%M")

    if base_jql.strip():
        return f"({base_jql}) AND updated >= \"{since_str}\""
    return f"updated >= \"{since_str}\""


def _adf_to_text(node: Any) -> str:
    """Convert Atlassian Document Format (ADF) nodes to plain text."""
    if node is None:
        return ""

    if isinstance(node, str):
        return node

    if isinstance(node, list):
        return "".join(_adf_to_text(item) for item in node)

    if not isinstance(node, dict):
        return ""

    node_type = node.get("type", "")

    if node_type == "text":
        return str(node.get("text", ""))

    content = node.get("content", [])
    text = "".join(_adf_to_text(item) for item in content)

    if node_type in {"paragraph", "heading", "listItem", "tableRow"}:
        return text + "\n"
    if node_type in {"bulletList", "orderedList", "table", "doc", "tableCell"}:
        return text

    return text


def _extract_comment_text(comment: Dict[str, Any]) -> str:
    """Extract plain text from Jira comment payloads (ADF or plain text)."""
    author_name = str(((comment.get("author") or {}).get("displayName", ""))).strip()
    body_raw = comment.get("body")
    body_text = _adf_to_text(body_raw).strip()

    if not body_text:
        return ""

    if author_name:
        return f"{author_name}: {body_text}"
    return body_text


def _issue_to_document(
    issue: Dict[str, Any],
    base_url: str,
    loader_id: str,
    include_comments: bool,
) -> Optional[Document]:
    """Map one Jira issue to one LangChain Document."""
    fields = issue.get("fields", {}) or {}

    issue_key = str(issue.get("key", "")).strip()
    issue_id = str(issue.get("id", "")).strip()

    if not issue_key and not issue_id:
        return None

    summary = str(fields.get("summary", "")).strip()
    description = _adf_to_text(fields.get("description", "")).strip()

    status_name = str(((fields.get("status") or {}).get("name", ""))).strip()
    assignee_name = str((((fields.get("assignee") or {}).get("displayName", "")))).strip()

    labels_raw = fields.get("labels", [])
    labels = [str(label) for label in labels_raw] if isinstance(labels_raw, list) else []

    updated = str(fields.get("updated", ""))
    created = str(fields.get("created", ""))

    project = fields.get("project") or {}
    project_key = str(project.get("key", "")).strip()
    project_name = str(project.get("name", "")).strip()

    comments_text: List[str] = []
    if include_comments:
        comment_container = fields.get("comment", {}) or {}
        comments = comment_container.get("comments", [])
        if isinstance(comments, list):
            comments_text = [text for text in (_extract_comment_text(c) for c in comments if isinstance(c, dict)) if text]

    page_parts: List[str] = []
    if issue_key:
        page_parts.append(f"Issue: {issue_key}")
    if summary:
        page_parts.append(f"Summary: {summary}")
    if description:
        page_parts.append(f"Description:\n{description}")
    if status_name:
        page_parts.append(f"Status: {status_name}")
    if assignee_name:
        page_parts.append(f"Assignee: {assignee_name}")
    if labels:
        page_parts.append(f"Labels: {', '.join(labels)}")
    if include_comments and comments_text:
        page_parts.append("Comments:\n" + "\n".join(comments_text))

    page_content = "\n\n".join(part for part in page_parts if part).strip()
    if not page_content:
        page_content = summary or issue_key or issue_id

    source = f"{base_url.rstrip('/')}/browse/{issue_key}" if issue_key else f"{base_url.rstrip('/')}/rest/api/3/issue/{issue_id}"
    content_hash = hashlib.sha256(page_content.encode("utf-8")).hexdigest()

    return Document(
        page_content=page_content,
        metadata={
            "source": source,
            "loader_id": loader_id,
            "loader": "JiraLoader",
            "content_hash": content_hash,
            "issue_key": issue_key,
            "issue_id": issue_id,
            "summary": summary,
            "status": status_name,
            "assignee": assignee_name,
            "labels": labels,
            "updated": updated,
            "created": created,
            "project_key": project_key,
            "project_name": project_name,
        },
    )


def _iter_issues(
    base_url: str,
    session: requests.Session,
    jql: str,
    page_size: int,
    fields: List[str],
) -> Generator[Dict[str, Any], None, None]:
    """Iterate all Jira issues for a query using paging and endpoint fallback."""
    endpoint_candidates = [
        f"{base_url.rstrip('/')}/rest/api/3/search",
        f"{base_url.rstrip('/')}/rest/api/2/search",
    ]

    endpoint: Optional[str] = None

    for candidate in endpoint_candidates:
        try:
            first_params: Dict[str, Union[str, int]] = {
                "jql": jql,
                "startAt": 0,
                "maxResults": page_size,
                "fields": ",".join(fields),
            }
            response = session.get(
                candidate,
                params=first_params,
                timeout=30,
            )
            if response.status_code in (404, 405):
                continue
            response.raise_for_status()
            first_data = response.json()
            endpoint = candidate

            issues = first_data.get("issues", [])
            total = int(first_data.get("total", len(issues)))

            for issue in issues:
                if isinstance(issue, dict):
                    yield issue

            start_at = int(first_data.get("startAt", 0)) + int(first_data.get("maxResults", len(issues)))

            while start_at < total:
                if stop_loading:
                    break

                next_params: Dict[str, Union[str, int]] = {
                    "jql": jql,
                    "startAt": start_at,
                    "maxResults": page_size,
                    "fields": ",".join(fields),
                }
                next_response = session.get(
                    endpoint,
                    params=next_params,
                    timeout=30,
                )
                next_response.raise_for_status()
                next_data = next_response.json()

                next_issues = next_data.get("issues", [])
                if not next_issues:
                    break

                for issue in next_issues:
                    if isinstance(issue, dict):
                        yield issue

                start_at += int(next_data.get("maxResults", len(next_issues)))

            return

        except requests.exceptions.RequestException as err:
            logger.warning("JiraLoader: endpoint %s failed: %s", candidate, err)
            continue

    if endpoint is None:
        logger.error("JiraLoader: no usable Jira search endpoint found under %s", base_url)


def load_from_jira(
    base_url: str,
    loader_id: str = "",
    auth_type: str = "basic",
    username: str = "",
    password: str = "",
    token: str = "",
    jql: str = "",
    projects: Optional[List[str]] = None,
    issue_types: Optional[List[str]] = None,
    page_size: int = 50,
    include_comments: bool = False,
    since: Optional[datetime] = None,
) -> List[Document]:
    """Load Jira issues and map them to one Document per issue."""
    projects = projects or []
    issue_types = issue_types or []

    effective_page_size = max(1, min(page_size, 100))

    base_jql = _build_base_jql(jql=jql, projects=projects, issue_types=issue_types)
    effective_jql = _add_since_to_jql(base_jql, since)

    fields = ["summary", "description", "status", "assignee", "labels", "updated", "created", "project"]
    if include_comments:
        fields.append("comment")

    session = _build_session(auth_type=auth_type, username=username, password=password, token=token)

    documents: List[Document] = []

    logger.info("JiraLoader: loading issues with JQL '%s'", effective_jql)
    for issue in _iter_issues(
        base_url=base_url,
        session=session,
        jql=effective_jql,
        page_size=effective_page_size,
        fields=fields,
    ):
        if stop_loading:
            logger.info("Loading process stopped by user.")
            break

        doc = _issue_to_document(
            issue=issue,
            base_url=base_url,
            loader_id=loader_id,
            include_comments=include_comments,
        )
        if doc is not None:
            documents.append(doc)

    logger.info("JiraLoader: loaded %s document(s)", len(documents))
    return documents


def get_all_jira_document_ids(
    base_url: str,
    auth_type: str = "basic",
    username: str = "",
    password: str = "",
    token: str = "",
    jql: str = "",
    projects: Optional[List[str]] = None,
    issue_types: Optional[List[str]] = None,
    page_size: int = 100,
) -> List[str]:
    """Return source IDs (issue URLs) for all Jira issues matching the query."""
    projects = projects or []
    issue_types = issue_types or []

    effective_page_size = max(1, min(page_size, 100))

    base_jql = _build_base_jql(jql=jql, projects=projects, issue_types=issue_types)

    session = _build_session(auth_type=auth_type, username=username, password=password, token=token)

    ids: List[str] = []
    for issue in _iter_issues(
        base_url=base_url,
        session=session,
        jql=base_jql,
        page_size=effective_page_size,
        fields=["key"],
    ):
        issue_key = str(issue.get("key", "")).strip()
        issue_id = str(issue.get("id", "")).strip()

        if issue_key:
            ids.append(f"{base_url.rstrip('/')}/browse/{issue_key}")
        elif issue_id:
            ids.append(f"{base_url.rstrip('/')}/rest/api/3/issue/{issue_id}")

    logger.info("get_all_jira_document_ids: total %s IDs", len(ids))
    return ids
