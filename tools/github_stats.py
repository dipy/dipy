#!/usr/bin/env python3
"""Simple tools to query github.com and gather stats about issues.

Taken from ipython

Usage
-----
List all issues/PRs since a tag (whole repo)::

    python3 tools/github_stats.py 1.12.0

List only PRs merged into a specific branch (for maintenance releases)::

    python3 tools/github_stats.py 1.12.0 --branch maint/1.12.x

"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import argparse
from datetime import datetime, timedelta
import json
import re
from subprocess import check_output
import sys
from urllib.parse import quote
from urllib.request import urlopen

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

ISO8601 = "%Y-%m-%dT%H:%M:%SZ"
PER_PAGE = 100

element_pat = re.compile(r'<(.+?)>')
rel_pat = re.compile(r'rel=[\'"](\w+)[\'"]')

LAST_RELEASE = datetime(2015, 3, 18)

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def parse_link_header(headers):
    link_s = headers.get('link', '')
    urls = element_pat.findall(link_s)
    rels = rel_pat.findall(link_s)
    d = {}
    for rel, url in zip(rels, urls):
        d[rel] = url
    return d


def get_paged_request(url):
    """Get a full list, handling APIv3's paging."""
    results = []
    while url:
        print(f"fetching {url}", file=sys.stderr)
        f = urlopen(url)
        results.extend(json.load(f))
        links = parse_link_header(f.headers)
        url = links.get('next')
    return results


def get_issues(project="dipy/dipy", state="closed", pulls=False):
    """Get a list of the issues from the Github API."""
    which = 'pulls' if pulls else 'issues'
    url = (
        f"https://api.github.com/repos/{project}/{which}"
        f"?state={state}&per_page={PER_PAGE}"
    )
    return get_paged_request(url)


def _parse_datetime(s):
    """Parse dates in the format returned by the Github API."""
    if s:
        return datetime.strptime(s, ISO8601)
    else:
        return datetime.fromtimestamp(0)


def issues2dict(issues):
    """Convert a list of issues to a dict, keyed by issue number."""
    idict = {}
    for i in issues:
        idict[i['number']] = i
    return idict


def is_pull_request(issue):
    """Return True if the given issue is a pull request."""
    return 'pull_request_url' in issue


def issues_closed_since(
    period=LAST_RELEASE, *, project="dipy/dipy", pulls=False, branch=None
):
    """Get all issues closed since a particular point in time.

    Parameters
    ----------
    period : datetime or timedelta
        If a timedelta, it is subtracted from the current time to get the
        cutoff datetime.
    project : str
        GitHub project in ``owner/repo`` format.
    pulls : bool
        If True fetch pull requests; otherwise fetch regular issues.
    branch : str or None
        When ``pulls=True`` and a branch name is given, only pull requests
        whose *base* (target) branch matches this value are returned.
        Has no effect when ``pulls=False``.

    Returns
    -------
    list
        Closed issues (or merged PRs) since *period*.
    """
    which = 'pulls' if pulls else 'issues'

    if isinstance(period, timedelta):
        period = datetime.now() - period
    url = (
        f"https://api.github.com/repos/{project}/{which}?state=closed"
        f"&sort=updated&since={period.strftime(ISO8601)}&per_page={PER_PAGE}"
    )
    if pulls and branch:
        url += f"&base={quote(branch, safe='')}"

    allclosed = get_paged_request(url)
    filtered = [i for i in allclosed
                if _parse_datetime(i['closed_at']) > period]

    if pulls:
        # exclude unmerged PRs
        filtered = [pr for pr in filtered if pr['merged_at']]
    else:
        # exclude pull requests from the issues list to avoid duplicates;
        # the /issues endpoint returns both issues and PRs
        filtered = [i for i in filtered if 'pull_request' not in i]

    return filtered


def sorted_by_field(issues, field='closed_at', reverse=False):
    """Return a list of issues sorted by closing date date."""
    return sorted(issues, key=lambda i: i[field], reverse=reverse)


def report(issues, show_urls=False):
    """Summary report about a list of issues, printing number and title."""
    # titles may have unicode in them, so we must encode everything below
    if show_urls:
        for i in issues:
            role = 'ghpull' if 'merged_at' in i else 'ghissue'
            print('* :%s:`%d`: %s' % (role, i['number'],
                                      i['title']))
    else:
        for i in issues:
            print('* %d: %s' % (i['number'], i['title']))

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Whether to add reST urls for all issues in printout.
    show_urls = True

    parser = argparse.ArgumentParser(
        description=(
            "Generate GitHub statistics for DIPY release notes.\n\n"
            "Examples:\n"
            "  # All issues/PRs since tag (master/full-repo release):\n"
            "  python3 tools/github_stats.py 1.12.0\n\n"
            "  # Only PRs merged into a maintenance branch:\n"
            "  python3 tools/github_stats.py 1.12.0 --branch maint/1.12.x\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "tag_or_days",
        nargs="?",
        metavar="TAG_OR_DAYS",
        help=(
            "A git tag (e.g. '1.12.0') to use as the start of the period, "
            "or an integer number of days to look back. "
            "Defaults to the most recent tag."
        ),
    )
    parser.add_argument(
        "--branch", "-b",
        default=None,
        metavar="BRANCH",
        help=(
            "Filter pull requests by base (target) branch. "
            "Use this for maintenance releases to avoid listing master PRs. "
            "Example: --branch maint/1.12.x"
        ),
    )
    args = parser.parse_args()

    tag = None
    days = None

    if args.tag_or_days is None:
        tag = check_output(['git', 'describe', '--abbrev=0'],
                           text=True).strip()
    else:
        try:
            days = int(args.tag_or_days)
        except ValueError:
            tag = args.tag_or_days

    if tag:
        cmd = ['git', 'log', '-1', '--format=%ai', tag]
        tagday, tz = check_output(cmd, text=True).strip().rsplit(' ', 1)
        since = datetime.strptime(tagday, "%Y-%m-%d %H:%M:%S")
    else:
        since = datetime.now() - timedelta(days=days)

    branch = args.branch

    print(f"fetching GitHub stats since {since} (tag: {tag})",
          file=sys.stderr)
    if branch:
        print(f"filtering pull requests by base branch: {branch}",
              file=sys.stderr)

    issues = issues_closed_since(since, pulls=False)
    pulls = issues_closed_since(since, pulls=True, branch=branch)

    # For regular reports, it's nice to show them in reverse
    # chronological order
    issues = sorted_by_field(issues, reverse=True)
    pulls = sorted_by_field(pulls, reverse=True)

    n_issues, n_pulls = map(len, (issues, pulls))
    n_total = n_issues + n_pulls

    # Print summary report we can directly include into release notes.
    print()
    since_day = since.strftime("%Y/%m/%d")
    today = datetime.today().strftime("%Y/%m/%d")
    print(f"GitHub stats for {since_day} - {today} (tag: {tag})")
    print()
    print("These lists are automatically generated, and may be incomplete or"
          " contain duplicates.")
    print()
    if tag:
        # print git info, in addition to GitHub info:
        since_tag = tag + '..'
        if branch:
            # limit git log to the specific branch when filtering
            cmd = ['git', 'log', '--oneline', since_tag, branch]
        else:
            cmd = ['git', 'log', '--oneline', since_tag]
        ncommits = len(check_output(cmd, text=True).splitlines())

        if branch:
            author_cmd = ['git', 'log', '--format=* %aN', since_tag, branch]
        else:
            author_cmd = ['git', 'log', '--format=* %aN', since_tag]
        all_authors = check_output(author_cmd, text=True).splitlines()
        unique_authors = sorted(set(all_authors))

        if not unique_authors:
            print("No commits during this period.")
        else:
            print(f"The following {len(unique_authors)} authors contributed"
                  f" {ncommits} commits.")
            print()
            print('\n'.join(unique_authors))
            print()

            print()
            if branch:
                print(
                    f"We closed a total of {n_pulls} pull requests"
                    f" (merged into ``{branch}``);\n"
                    "this is the full list (generated with the script \n"
                    ":file:`tools/github_stats.py`):"
                )
            else:
                print(
                    f"We closed a total of {n_total} issues,"
                    f" {n_pulls} pull requests and {n_issues} regular issues;\n"
                    "this is the full list (generated with the script \n"
                    ":file:`tools/github_stats.py`):"
                )
            print()
            print(f'Pull Requests ({n_pulls}):\n')
            report(pulls, show_urls)
            if not branch:
                print()
                print(f'Issues ({n_issues}):\n')
                report(issues, show_urls)
