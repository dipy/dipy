#!/usr/bin/env python3
"""Simple tools to query github.com and gather stats about issues.

Taken from ipython

"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from datetime import datetime, timedelta
import json
import re
from subprocess import check_output
import sys
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


def issues_closed_since(period=LAST_RELEASE, project="dipy/dipy", pulls=False):
    """Get all issues closed since a particular point in time.

    Period can either be a datetime object, or a timedelta object. In the
    latter case, it is used as a time before the present.

    """
    which = 'pulls' if pulls else 'issues'

    if isinstance(period, timedelta):
        period = datetime.now() - period
    url = (
        f"https://api.github.com/repos/{project}/{which}?state=closed"
        f"&sort=updated&since={period.strftime(ISO8601)}&per_page={PER_PAGE}"
    )

    allclosed = get_paged_request(url)
    # allclosed = get_issues(project=project, state='closed', pulls=pulls,
    #                        since=period)
    filtered = [i for i in allclosed
                if _parse_datetime(i['closed_at']) > period]

    # exclude rejected PRs
    if pulls:
        filtered = [pr for pr in filtered if pr['merged_at']]

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

    # By default, search one month back
    tag = None
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except:
            tag = sys.argv[1]
    else:
        tag = check_output(['git', 'describe', '--abbrev=0'],
                           text=True).strip()

    if tag:
        cmd = ['git', 'log', '-1', '--format=%ai', tag]
        tagday, tz = check_output(cmd, text=True).strip().rsplit(' ', 1)
        since = datetime.strptime(tagday, "%Y-%m-%d %H:%M:%S")
    else:
        since = datetime.now() - timedelta(days=days)

    print(f"fetching GitHub stats since {since} (tag: {tag})",
          file=sys.stderr)
    # turn off to play interactively without redownloading, use %run -i
    if 1:
        issues = issues_closed_since(since, pulls=False)
        pulls = issues_closed_since(since, pulls=True)

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
        cmd = ['git', 'log', '--oneline', since_tag]
        ncommits = len(check_output(cmd, text=True).splitlines())

        author_cmd = ['git', 'log', '--format=* %aN', since_tag]
        all_authors = check_output(author_cmd, text=True) \
            .splitlines()
        unique_authors = sorted(set(all_authors))

        if not unique_authors:
            print("No commits during this period.")
        else:
            print(f"The following {len(unique_authors)} authors contributed {ncommits} commits.")
            print()
            print('\n'.join(unique_authors))
            print()

            print()
            print(f"We closed a total of {n_total} issues,"
                  f" {n_pulls} pull requests and {n_issues} regular issues;\n"
                  "this is the full list (generated with the script \n"
                  ":file:`tools/github_stats.py`):")
            print()
            print(f'Pull Requests ({n_pulls}):\n')
            report(pulls, show_urls)
            print()
            print(f'Issues ({n_issues}):\n')
            report(issues, show_urls)
