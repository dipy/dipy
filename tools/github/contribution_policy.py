"""Enforce the DIPY contribution policy from GitHub Actions.

Only the Python standard library is used, so the workflows install nothing.
Each subcommand is called by a workflow in ``.github/workflows``:

- ``scan-pull-requests``: for every open pull request updated in the last
  ``--window`` minutes, close it if it comes from an early contributor and
  closes an issue still labeled ``type:needs triage``, and otherwise comment
  when its "AI Usage Disclosure" section is missing or not filled.
- ``autoclose``: warn pull requests labeled ``autoclose`` and close them once
  the label is older than ``AUTOCLOSE_DAYS`` days.
- ``sync-triage-warning``: label blank issues with ``type:needs triage`` and
  add or remove the "not ready for a pull request" banner of the issue given
  by ``NUMBER``.

The workflows provide ``GITHUB_TOKEN`` and ``GITHUB_REPOSITORY``.
"""

import argparse
from datetime import UTC, datetime, timedelta
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

API_URL = "https://api.github.com"
API_VERSION = "2022-11-28"

TRIAGE_LABEL = "type:needs triage"
NOT_READY_LABEL = "linked issue not ready"
AUTOCLOSE_LABEL = "autoclose"
AUTOCLOSE_DAYS = 14
MIN_MERGED_PRS = 5
TRUSTED_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

CONTRIBUTING_URL = "https://github.com/dipy/dipy/blob/master/.github/CONTRIBUTING.md"
TRIAGE_URL = f"{CONTRIBUTING_URL}#issues-labeled-typeneeds-triage"
AI_POLICY_URL = f"{CONTRIBUTING_URL}#automated-contributions-policy"
PR_TEMPLATE_URL = (
    "https://github.com/dipy/dipy/blob/master/.github/PULL_REQUEST_TEMPLATE.md"
)

AI_SECTION_HEADING = "AI Usage Disclosure"
NO_AI_OPTION = "I did not use any AI tool"
AI_USED_OPTION = "I used AI assistance for"
AI_TOOL_LINE = re.compile(r"AI tool\(s\) used:[ \t]*(\S.*)?$", re.MULTILINE)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
AI_CHECK_MARKER = "<!-- dipy-ai-disclosure-check -->"
AUTOCLOSE_MARKER = "<!-- dipy-autoclose-warning -->"

TRIAGE_BANNER = (
    "> [!WARNING]\n"
    "> This issue is not yet ready for a pull request. If you are interested in "
    "contributing to DIPY, please read our [contributing guidelines]"
    f"({CONTRIBUTING_URL}), in particular the section on the "
    f'["type:needs triage"]({TRIAGE_URL}) label.'
)

LINKED_ISSUE_NOT_READY_MESSAGE = (
    "Thank you for your interest in contributing to DIPY.\n\n"
    f"The linked issue is still labeled `{TRIAGE_LABEL}`: the maintainers have "
    "not yet reached a consensus on how it should be resolved. Before opening a "
    "pull request, please read the "
    f'["Issues labeled type:needs triage"]({TRIAGE_URL}) section of the '
    "contributing guidelines.\n\n"
    "For now, we are closing this pull request.\n\n"
    "* If you believe your proposed change addresses the linked issue, please "
    "explain your proposal in that issue and allow time for discussion with the "
    "maintainers, so that a consensus can be reached before implementation "
    "begins.\n"
    "* If you believe this is a mistake, please leave a comment on the linked "
    "issue."
)

AI_DISCLOSURE_PROBLEMS = {
    "missing": (
        f'The "{AI_SECTION_HEADING}" section of the pull request template is '
        "missing from the description."
    ),
    "empty": f'The "{AI_SECTION_HEADING}" section is empty.',
    "unedited": (
        f'The "{AI_SECTION_HEADING}" section still contains all the options of '
        "the template. Keep the items that apply and delete the rest."
    ),
    "no_tool": (
        "The description says AI assistance was used, but does not name the AI tool(s)."
    ),
}

AI_DISCLOSURE_REQUEST = (
    f"{AI_CHECK_MARKER}\n"
    "Thank you for your pull request.\n\n"
    "{problem}\n\n"
    "Disclosing AI usage is mandatory for every pull request to DIPY, including "
    "when no AI tool was used. Please edit the description of this pull request "
    f'and fill the "{AI_SECTION_HEADING}" section of the [pull request template]'
    f"({PR_TEMPLATE_URL}). See our [Automated Contributions Policy]"
    f"({AI_POLICY_URL}). This comment updates itself once the section is filled."
)

AI_DISCLOSURE_RESOLVED = (
    f"{AI_CHECK_MARKER}\n"
    f'Thank you, the "{AI_SECTION_HEADING}" section is now filled. '
    ":white_check_mark:"
)

AUTOCLOSE_WARNING = (
    f"{AUTOCLOSE_MARKER}\n"
    f":alarm_clock: This pull request might be automatically closed in "
    f"{AUTOCLOSE_DAYS} days from now.\n\n"
    "Thank you for your contribution to DIPY and for the effort you have put into "
    "this pull request. It does not yet meet the quality and clarity needed for an "
    "effective review. Maintainers have limited time for code reviews, and our "
    "goal is to prioritize well-prepared contributions to keep DIPY maintainable."
    "\n\n"
    "As the author, you are responsible for driving this pull request: doing the "
    "necessary background research and presenting its context and your thought "
    "process. Please read our [contributing guidelines]"
    f"({CONTRIBUTING_URL}), including the [Automated Contributions Policy]"
    f"({AI_POLICY_URL}).\n\n"
    f"If you substantially improve this pull request within {AUTOCLOSE_DAYS} days, "
    f"a maintainer may remove the `{AUTOCLOSE_LABEL}` label and it stays open. "
    "Cosmetic changes or incomplete fixes will not be sufficient. Please do not "
    "ping (`@`) maintainers."
)

AUTOCLOSE_MESSAGE = (
    "Thank you for your interest in contributing to DIPY. We cannot accept this "
    "pull request as it does not meet our development standards.\n\n"
    f"Following our autoclose policy, we are closing it after allowing "
    f"{AUTOCLOSE_DAYS} days for improvements. If you think it has been closed by "
    "mistake, please comment below."
)

LINKED_ISSUES_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      closingIssuesReferences(first: 10) {
        nodes {
          number
          labels(first: 50) { nodes { name } }
        }
      }
    }
  }
}
"""

NEXT_LINK = re.compile(r'<([^>]+)>;\s*rel="next"')


def extract_section(*, body, heading):
    """Return the content of a level-2 markdown section.

    Parameters
    ----------
    body : str
        Markdown text.
    heading : str
        Title of the ``##`` section, matched case-insensitively.

    Returns
    -------
    str or None
        Text between the heading and the next ``##`` heading, or None when the
        heading is absent.
    """
    pattern = re.compile(
        rf"^##\s*{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(body)
    return None if match is None else match.group(1)


def check_ai_disclosure(*, body):
    """Check the "AI Usage Disclosure" section of a pull request description.

    Parameters
    ----------
    body : str or None
        Pull request description.

    Returns
    -------
    str or None
        Key of ``AI_DISCLOSURE_PROBLEMS`` describing the problem, or None when
        the section is filled.
    """
    section = extract_section(body=body or "", heading=AI_SECTION_HEADING)
    if section is None:
        return "missing"
    section = HTML_COMMENT.sub("", section).strip()
    if not section:
        return "empty"
    no_ai = NO_AI_OPTION.lower() in section.lower()
    ai_used = AI_USED_OPTION.lower() in section.lower()
    if no_ai and ai_used:
        return "unedited"
    tool = AI_TOOL_LINE.search(section)
    if ai_used and (tool is None or tool.group(1) is None):
        return "no_tool"
    return None


def add_banner(*, body):
    """Prepend the triage banner to an issue description.

    Parameters
    ----------
    body : str
        Issue description.

    Returns
    -------
    str
        Description starting with ``TRIAGE_BANNER``, unchanged if it already
        does.
    """
    if body.startswith(TRIAGE_BANNER):
        return body
    return f"{TRIAGE_BANNER}\n\n{body}"


def remove_banner(*, body):
    """Remove the triage banner from an issue description.

    Parameters
    ----------
    body : str
        Issue description.

    Returns
    -------
    str
        Description without the leading ``TRIAGE_BANNER``.
    """
    return body.removeprefix(TRIAGE_BANNER).lstrip("\n")


def is_expired(*, labeled_at, now, days):
    """Tell whether a label was added more than ``days`` days ago.

    Parameters
    ----------
    labeled_at : datetime or None
        Time the label was last added, None if never.
    now : datetime
        Current time.
    days : int
        Grace period in days.

    Returns
    -------
    bool
        True when the grace period is over.
    """
    return labeled_at is not None and now - labeled_at > timedelta(days=days)


def is_bot(*, user):
    """Tell whether a GitHub user is a bot account.

    Parameters
    ----------
    user : dict
        ``user`` object of a pull request or issue from the REST API.

    Returns
    -------
    bool
        True for bot accounts such as dependabot.
    """
    return user["type"] == "Bot" or user["login"].endswith("[bot]")


def parse_time(*, value):
    """Parse a timestamp of the GitHub API.

    Parameters
    ----------
    value : str
        ISO 8601 timestamp such as ``2026-10-26T12:00:00Z``.

    Returns
    -------
    datetime
        Timezone-aware datetime.
    """
    return datetime.fromisoformat(value)


def request(*, method, url, data=None):
    """Send an authenticated request to the GitHub API.

    Parameters
    ----------
    method : str
        HTTP method.
    url : str
        Path relative to ``API_URL``, or absolute URL.
    data : dict, optional
        JSON payload.

    Returns
    -------
    payload : dict or list or None
        Decoded JSON response, None when the response is empty.
    next_url : str or None
        URL of the next page of a paginated response.
    """
    if not url.startswith("https://"):
        url = f"{API_URL}{url}"
    req = urllib.request.Request(
        url,
        method=method,
        data=None if data is None else json.dumps(data).encode(),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Content-Type": "application/json",
            "User-Agent": "dipy-contribution-policy",
            "X-GitHub-Api-Version": API_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
            link = NEXT_LINK.search(response.headers.get("Link", ""))
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{method} {url} failed: {error.code} {detail}") from error
    return (json.loads(raw) if raw else None), (link.group(1) if link else None)


def api(*, method, url, data=None):
    """Send a request to the GitHub API and return the decoded response.

    Parameters
    ----------
    method : str
        HTTP method.
    url : str
        Path relative to ``API_URL``.
    data : dict, optional
        JSON payload.

    Returns
    -------
    dict or list or None
        Decoded JSON response.
    """
    return request(method=method, url=url, data=data)[0]


def paginate(*, url):
    """Return all the items of a paginated GitHub API listing.

    Parameters
    ----------
    url : str
        Path relative to ``API_URL``, with its query string.

    Returns
    -------
    list
        Items of every page. For the search API, the ``items`` of every page.
    """
    items = []
    while url:
        payload, url = request(method="GET", url=url)
        items.extend(payload["items"] if isinstance(payload, dict) else payload)
    return items


def search(*, query):
    """Run an issue and pull request search.

    Parameters
    ----------
    query : str
        GitHub search query.

    Returns
    -------
    str
        Path of the first page of results.
    """
    return f"/search/issues?{urllib.parse.urlencode({'q': query, 'per_page': 100})}"


def repo_url():
    """Return the API path of the repository of the running workflow.

    Returns
    -------
    str
        ``/repos/<owner>/<name>``.
    """
    return f"/repos/{os.environ['GITHUB_REPOSITORY']}"


def comment(*, number, body):
    """Post a comment on an issue or pull request.

    Parameters
    ----------
    number : int
        Issue or pull request number.
    body : str
        Markdown comment.
    """
    api(
        method="POST",
        url=f"{repo_url()}/issues/{number}/comments",
        data={"body": body},
    )


def close(*, number):
    """Close an issue or pull request.

    Parameters
    ----------
    number : int
        Issue or pull request number.
    """
    api(method="PATCH", url=f"{repo_url()}/issues/{number}", data={"state": "closed"})


def list_comments(*, number):
    """Return the comments of an issue or pull request.

    Parameters
    ----------
    number : int
        Issue or pull request number.

    Returns
    -------
    list of dict
        Comments, oldest first.
    """
    return paginate(url=f"{repo_url()}/issues/{number}/comments?per_page=100")


def get_linked_issues(*, number):
    """Return the issues a pull request closes, with their labels.

    Parameters
    ----------
    number : int
        Pull request number.

    Returns
    -------
    list of dict
        GraphQL nodes with ``number`` and ``labels`` keys.
    """
    owner, name = os.environ["GITHUB_REPOSITORY"].split("/")
    payload = api(
        method="POST",
        url="/graphql",
        data={
            "query": LINKED_ISSUES_QUERY,
            "variables": {"owner": owner, "name": name, "number": number},
        },
    )
    if payload.get("errors"):
        raise RuntimeError(f"GraphQL query failed: {payload['errors']}")
    pr = payload["data"]["repository"]["pullRequest"]
    return pr["closingIssuesReferences"]["nodes"]


def count_merged_prs(*, login):
    """Count the merged pull requests of a user in the repository.

    Parameters
    ----------
    login : str
        GitHub login.

    Returns
    -------
    int
        Number of merged pull requests.
    """
    query = f"repo:{os.environ['GITHUB_REPOSITORY']} is:pr is:merged author:{login}"
    return api(method="GET", url=search(query=query))["total_count"]


def enforce_linked_issue(*, pr):
    """Close a pull request that closes an issue still needing triage.

    Parameters
    ----------
    pr : dict
        Pull request, as an item of the search API.

    Returns
    -------
    bool
        True when the pull request was closed.
    """
    if pr["author_association"] in TRUSTED_ASSOCIATIONS:
        return False
    not_ready = [
        issue["number"]
        for issue in get_linked_issues(number=pr["number"])
        if any(label["name"] == TRIAGE_LABEL for label in issue["labels"]["nodes"])
    ]
    if not not_ready or count_merged_prs(login=pr["user"]["login"]) >= (MIN_MERGED_PRS):
        return False
    print(f"Closing #{pr['number']}: linked issue(s) {not_ready} need triage")
    comment(number=pr["number"], body=LINKED_ISSUE_NOT_READY_MESSAGE)
    close(number=pr["number"])
    api(
        method="POST",
        url=f"{repo_url()}/issues/{pr['number']}/labels",
        data={"labels": [NOT_READY_LABEL]},
    )
    return True


def enforce_ai_disclosure(*, pr):
    """Post, update or resolve the AI disclosure comment of a pull request.

    Parameters
    ----------
    pr : dict
        Pull request, as an item of the search API.
    """
    if is_bot(user=pr["user"]):
        return
    problem = check_ai_disclosure(body=pr["body"])
    previous = next(
        (
            c
            for c in list_comments(number=pr["number"])
            if c["body"].startswith(AI_CHECK_MARKER)
        ),
        None,
    )
    if problem is None:
        message = None if previous is None else AI_DISCLOSURE_RESOLVED
    else:
        print(f"#{pr['number']}: AI disclosure problem '{problem}'")
        message = AI_DISCLOSURE_REQUEST.format(problem=AI_DISCLOSURE_PROBLEMS[problem])
    if message is None or (previous is not None and previous["body"] == message):
        return
    if previous is None:
        comment(number=pr["number"], body=message)
    else:
        api(
            method="PATCH",
            url=f"{repo_url()}/issues/comments/{previous['id']}",
            data={"body": message},
        )


def run_each(*, items, action):
    """Apply an action to every item, reporting failures at the end.

    Parameters
    ----------
    items : list of dict
        Issues or pull requests with a ``number`` key.
    action : callable
        Called as ``action(item=item)``.
    """
    failed = []
    for item in items:
        try:
            action(item=item)
        except RuntimeError as error:
            print(f"#{item['number']}: {error}", file=sys.stderr)
            failed.append(item["number"])
    if failed:
        sys.exit(f"Failed on {failed}")


def scan_pull_requests(*, window):
    """Check the open pull requests updated in the last ``window`` minutes.

    Parameters
    ----------
    window : int
        Look-back period in minutes. It should exceed the interval of the
        schedule, so that a delayed or skipped run does not miss a change.
    """
    since = datetime.now(UTC) - timedelta(minutes=window)
    query = (
        f"repo:{os.environ['GITHUB_REPOSITORY']} is:pr is:open "
        f"updated:>={since.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    prs = paginate(url=search(query=query))
    print(f"Checking {len(prs)} pull request(s) updated since {since:%H:%M} UTC")

    def check(*, item):
        if not enforce_linked_issue(pr=item):
            enforce_ai_disclosure(pr=item)

    run_each(items=prs, action=check)


def last_labeled_at(*, number, label):
    """Return when a label was last added to an issue or pull request.

    Parameters
    ----------
    number : int
        Issue or pull request number.
    label : str
        Label name.

    Returns
    -------
    datetime or None
        Time of the last ``labeled`` event for ``label``.
    """
    events = paginate(url=f"{repo_url()}/issues/{number}/events?per_page=100")
    times = [
        parse_time(value=event["created_at"])
        for event in events
        if event["event"] == "labeled" and event["label"]["name"] == label
    ]
    return max(times, default=None)


def autoclose():
    """Warn pull requests labeled ``autoclose`` and close the expired ones."""
    now = datetime.now(UTC)
    labeled = paginate(
        url=f"{repo_url()}/issues?"
        + urllib.parse.urlencode(
            {"state": "open", "labels": AUTOCLOSE_LABEL, "per_page": 100}
        )
    )
    prs = [item for item in labeled if "pull_request" in item]
    print(f"Found {len(prs)} open pull request(s) labeled {AUTOCLOSE_LABEL}")

    def handle(*, item):
        number = item["number"]
        labeled_at = last_labeled_at(number=number, label=AUTOCLOSE_LABEL)
        if is_expired(labeled_at=labeled_at, now=now, days=AUTOCLOSE_DAYS):
            print(f"Closing #{number}: {item['title']}")
            comment(number=number, body=AUTOCLOSE_MESSAGE)
            close(number=number)
            return
        warned = any(
            c["body"].startswith(AUTOCLOSE_MARKER)
            and labeled_at is not None
            and parse_time(value=c["created_at"]) >= labeled_at
            for c in list_comments(number=number)
        )
        if not warned:
            print(f"Warning #{number}")
            comment(number=number, body=AUTOCLOSE_WARNING)

    run_each(items=prs, action=handle)


def sync_triage_warning():
    """Label blank issues and keep the triage banner in sync with the label."""
    number = int(os.environ["NUMBER"])
    issue = api(method="GET", url=f"{repo_url()}/issues/{number}")
    labels = {label["name"] for label in issue["labels"]}
    if not labels and os.environ.get("EVENT_ACTION") == "opened":
        api(
            method="POST",
            url=f"{repo_url()}/issues/{number}/labels",
            data={"labels": [TRIAGE_LABEL]},
        )
        labels.add(TRIAGE_LABEL)
    body = issue["body"] or ""
    new_body = (
        add_banner(body=body) if TRIAGE_LABEL in labels else remove_banner(body=body)
    )
    if new_body != body:
        api(
            method="PATCH", url=f"{repo_url()}/issues/{number}", data={"body": new_body}
        )
        print(f"Updated triage banner of #{number}")


def main():
    """Run the subcommand given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan-pull-requests")
    scan.add_argument("--window", type=int, default=35, help="minutes")
    commands.add_parser("autoclose")
    commands.add_parser("sync-triage-warning")
    args = parser.parse_args()
    if args.command == "scan-pull-requests":
        scan_pull_requests(window=args.window)
    elif args.command == "autoclose":
        autoclose()
    else:
        sync_triage_warning()


if __name__ == "__main__":
    main()
