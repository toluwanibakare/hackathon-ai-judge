#!/usr/bin/env python3
"""Hackathon AI judge.

Ranks public GitHub repos against a set of judging criteria and picks the top 3.

How it works:
  1. Reads one repo URL per line from a file (or from the command line).
  2. Fetches each repo's README and metadata from the GitHub API.
     Public repos need no key. Set GITHUB_TOKEN to raise the rate limit.
  3. Scores every repo against the criteria in a criteria JSON file.
     With --llm groq (the default) and a GROQ_API_KEY set, an LLM scores
     every criterion with evidence quoted from the README. Gemini is also
     available with --llm gemini and GEMINI_API_KEY. Without a key, a
     transparent rule based pre-screen scores only what can be measured
     (README quality, demo evidence) and leaves usefulness and creativity
     to the human judges.
  4. Prints a ranked table, writes a Markdown report and a JSON file.

Usage:
  python3 judge.py --repos repos.txt --criteria criteria.json --out results.md
  python3 judge.py https://github.com/owner/repo --llm none
  python3 judge.py --submissions responses.csv --llm groq --out results.md

Submissions mode reads a form export CSV (team, members, project, link,
what it does, who it is for, how to use it), checks each project link, and
scores each entry against the same criteria. No GitHub account needed.

The AI judge proposes. Human judges decide. Always verify the top 3 by
actually trying their demos before announcing winners.
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
import urllib.error
import urllib.request

GITHUB_API = "https://api.github.com"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def gh_get(path, token=None):
    req = urllib.request.Request(
        GITHUB_API + path,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "hackathon-ai-judge"},
    )
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        if err.code == 403:
            raise SystemExit("GitHub rate limit reached. Set GITHUB_TOKEN to raise the limit.")
        raise


def parse_repo(url):
    url = url.strip()
    match = re.search(r"github\.com/([^/\s#?]+)/([^/\s#?]+)", url)
    if not match:
        match = re.match(r"^([^/\s]+)/([^/\s]+)$", url)
    if not match:
        raise ValueError("Cannot parse repo from: " + url)
    owner, repo = match.group(1), match.group(2)
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def fetch_repo(url, token):
    owner, repo = parse_repo(url)
    meta = gh_get("/repos/{}/{}".format(owner, repo), token)
    if meta is None:
        return None
    readme = gh_get("/repos/{}/{}/readme".format(owner, repo), token)
    text = ""
    if readme and readme.get("content"):
        text = base64.b64decode(readme["content"]).decode("utf-8", "replace")
    contents = gh_get("/repos/{}/{}/contents".format(owner, repo), token) or []
    files = [c["name"] for c in contents if isinstance(c, dict)]
    return {
        "url": "https://github.com/{}/{}".format(owner, repo),
        "name": meta.get("full_name"),
        "description": meta.get("description") or "",
        "homepage": meta.get("homepage") or "",
        "language": meta.get("language"),
        "pushed_at": meta.get("pushed_at"),
        "readme": text[:12000],
        "readme_truncated": len(text) > 12000,
        "files": files,
        "file_count": len(files),
    }


def rule_scores(repo):
    """Transparent pre-screen: only what can be measured without an LLM."""
    readme = repo["readme"]
    low = readme.lower()
    presentation = 0
    if readme:
        presentation += 3
    if len(readme) > 500:
        presentation += 2
    if len(readme) > 2000:
        presentation += 1
    for keyword in ("install", "usage", "demo", "screenshot"):
        if keyword in low:
            presentation += 1
    demo = 0
    if repo["homepage"]:
        demo += 4
    if re.search(r"demo|live|vercel|netlify|github\.io|youtube|youtu\.be|loom", low):
        demo += 3
    if "screenshot" in low or ".png" in low or ".jpg" in low:
        demo += 2
    if repo["file_count"] > 3:
        demo += 1
    return {
        "presentation": min(10, presentation),
        "working_demo": min(10, demo),
    }


def build_prompt(repo, criteria):
    crit_text = "\n".join(
        "- {} (weight {}): {}".format(c["key"], c["weight"], c["description"])
        for c in criteria
    )
    return (
        "You are a hackathon judge. Score this project against each criterion from 0 to 10.\n"
        "Base every score on evidence in the README or repo data below. If the README does not "
        "prove something, score it low and say so. Do not reward a long README alone.\n\n"
        "Criteria:\n" + crit_text + "\n\n"
        "Project: {}\nDescription: {}\nHomepage: {}\nMain language: {}\nFiles: {}\n\nREADME:\n{}\n\n"
        "Reply with strict JSON only, no other text:\n"
        '{{"scores": {{"<criterion key>": {{"score": <0-10>, "evidence": "<one line>"}}}}}}, '
        '"summary": "<one sentence>"}}'
    ).format(
        repo["name"], repo["description"], repo["homepage"], repo["language"],
        ", ".join(repo["files"][:40]), repo["readme"] or "(no README found)",
    )


def call_gemini(prompt, api_key, model):
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    url = GEMINI_URL.format(model=model) + "?key=" + api_key
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        result = json.loads(resp.read().decode())
    return result["candidates"][0]["content"]["parts"][0]["text"]


def call_groq(prompt, api_key, model):
    body = json.dumps({
        "model": model,
        "temperature": 0.2,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        GROQ_URL, data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        result = json.loads(resp.read().decode())
    return result["choices"][0]["message"]["content"]


def llm_scores(repo, criteria, provider, api_key, model):
    """Score via the chosen LLM provider. Returns parsed dict or None on failure."""
    prompt = build_prompt(repo, criteria)
    try:
        if provider == "groq":
            text = call_groq(prompt, api_key, model)
        else:
            text = call_gemini(prompt, api_key, model)
    except urllib.error.HTTPError as err:
        print("{} request failed (HTTP {}), falling back to pre-screen.".format(provider, err.code), file=sys.stderr)
        return None
    except (KeyError, IndexError, ValueError):
        print("Could not read {} reply, falling back to pre-screen.".format(provider), file=sys.stderr)
        return None
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        return json.loads(text[start:end])
    except ValueError:
        print("Could not parse {} reply, falling back to pre-screen.".format(provider), file=sys.stderr)
        return None


def rule_submission_scores(sub):
    """Transparent pre-screen for form submissions."""
    what, who, how = sub["what"], sub["who"], sub["how"]
    presentation = 0
    if what:
        presentation += 3
    if who:
        presentation += 2
    if how:
        presentation += 2
    if len(what) + len(who) + len(how) > 200:
        presentation += 1
    if sub["link"]:
        presentation += 1
    if sub["link_ok"]:
        presentation += 1
    ease = 0
    if how:
        ease += 5
    if re.search(r"step|\d\.|-|•", how.lower()):
        ease += 3
    if sub["link_ok"]:
        ease += 2
    return {
        "presentation": min(10, presentation),
        "ease_of_use": min(10, ease),
    }


def apply_scores(llm_project, measure_obj, criteria, llm, api_key, model, measured):
    """Score one project. LLM mode scores every criterion; without a key the
    rule based pre-screen scores only what can be measured."""
    scores, evidence = {}, {}
    if llm in ("gemini", "groq") and api_key:
        judged = llm_scores(llm_project, criteria, llm, api_key, model)
        if judged:
            for crit in criteria:
                entry = judged.get("scores", {}).get(crit["key"], {})
                try:
                    scores[crit["key"]] = max(0, min(10, float(entry.get("score", 0))))
                except (TypeError, ValueError):
                    scores[crit["key"]] = 0
                evidence[crit["key"]] = entry.get("evidence", "")
            return scores, evidence, [judged.get("summary", "")], "llm"
    measured_scores = measured(measure_obj)
    for crit in criteria:
        if crit["key"] in measured_scores:
            scores[crit["key"]] = measured_scores[crit["key"]]
            evidence[crit["key"]] = "Measured from the submission."
        else:
            scores[crit["key"]] = None
            evidence[crit["key"]] = "Needs a human judge or LLM scoring."
    return scores, evidence, [], "pre-screen"


def score_repo(repo, criteria, llm, api_key, model):
    scores, evidence, notes, mode = apply_scores(repo, repo, criteria, llm, api_key, model, rule_scores)
    if not repo["readme"]:
        notes.append("No README found, so presentation scores 0.")
    if repo["readme_truncated"]:
        notes.append("README truncated to 12000 characters for scoring.")
    return scores, evidence, notes, mode


def normalize_header(header):
    return re.sub(r"[^a-z0-9]", "", (header or "").lower())


def read_submissions(path):
    """Read a form export CSV. Columns are matched by header wording, so
    Google Forms question text works as-is. Returns a list of dicts."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    field_map = {}
    for raw in rows[0].keys():
        norm = normalize_header(raw)
        if "link" in norm or "url" in norm:
            field_map["link"] = raw
        elif "member" in norm:
            field_map["members"] = raw
        elif "what" in norm or "description" in norm or "does" in norm:
            field_map["what"] = raw
        elif "who" in norm or "audience" in norm:
            field_map["who"] = raw
        elif "how" in norm or "step" in norm or "use" in norm:
            field_map["how"] = raw
        elif "project" in norm:
            field_map["project"] = raw
        elif "team" in norm:
            field_map["team"] = raw
    submissions = []
    for index, row in enumerate(rows, 1):
        def get(key):
            return (row.get(field_map[key]) or "").strip() if key in field_map else ""
        link = get("link")
        if link and not re.match(r"https?://", link):
            link = "https://" + link
        submissions.append({
            "team": get("team") or "Team {}".format(index),
            "members": get("members"),
            "project": get("project") or "Untitled project",
            "link": link,
            "what": get("what"),
            "who": get("who"),
            "how": get("how"),
        })
    return submissions


def check_link(url):
    """Returns (ok, status_text). Tries HEAD, then a short GET, because some
    hosts reject HEAD but serve GET fine. A failed check flags the entry, it
    never disqualifies it on its own."""
    if not url:
        return False, "no link provided"
    agent = {"User-Agent": "Mozilla/5.0 (hackathon-ai-judge)"}
    try:
        req = urllib.request.Request(url, method="HEAD", headers=agent)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return True, "HTTP {}".format(resp.status)
    except ValueError:
        return False, "invalid link"
    except (urllib.error.HTTPError, urllib.error.URLError):
        pass
    try:
        req = urllib.request.Request(url, method="GET", headers=agent)
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read(1)
            return True, "HTTP {}".format(resp.status)
    except urllib.error.HTTPError as err:
        return False, "HTTP {}".format(err.code)
    except (urllib.error.URLError, ValueError):
        return False, "unreachable"


def score_submission(sub, criteria, llm, api_key, model):
    combined = "What it does: {}\nWho it is for: {}\nHow to use it: {}\nLink check: {}".format(
        sub["what"] or "(not provided)", sub["who"] or "(not provided)",
        sub["how"] or "(not provided)", sub["link_status"])
    project = {
        "name": "{} ({})".format(sub["project"], sub["team"]),
        "description": combined, "homepage": sub["link"], "language": "",
        "files": [], "readme": combined, "link_ok": sub["link_ok"],
    }
    scores, evidence, notes, mode = apply_scores(
        project, sub, criteria, llm, api_key, model, rule_submission_scores)
    notes.append("Link: {} ({})".format(sub["link"] or "none", sub["link_status"]))
    if not sub["what"]:
        notes.append("No project description provided.")
    return scores, evidence, notes, mode


def run_submissions(args, criteria, llm, api_key, model):
    subs = read_submissions(args.submissions)
    if not subs:
        raise SystemExit("No submissions found in {}.".format(args.submissions))
    results = []
    for sub in subs:
        sub["link_ok"], sub["link_status"] = check_link(sub["link"])
        scores, evidence, notes, mode = score_submission(sub, criteria, llm, api_key, model)
        scored = [(c["weight"], scores[c["key"]]) for c in criteria if scores[c["key"]] is not None]
        points = round(sum(w * s for w, s in scored) / 10, 1) if scored else 0
        max_points = sum(w for w, _ in scored)
        ratio = (points / max_points) if max_points else 0
        results.append({
            "sub": sub, "scores": scores, "evidence": evidence,
            "notes": notes, "mode": mode, "points": points,
            "max_points": max_points, "ratio": ratio,
        })
    results.sort(key=lambda r: r["ratio"], reverse=True)

    print("\nRanked submissions (points out of measured criteria only):")
    for rank, item in enumerate(results, 1):
        mark = "  <-- TOP {}".format(rank) if rank <= args.top else ""
        print("{}. {} - {} ({}/{}){}".format(
            rank, item["sub"]["team"], item["sub"]["project"],
            item["points"], item["max_points"], mark))
    print("\nWinners:")
    for rank, item in enumerate(results[: args.top], 1):
        print("  {}. {} - {}".format(rank, item["sub"]["team"], item["sub"]["project"]))

    lines = ["# Hackathon judging results (form submissions)\n"]
    lines.append("Scored against: " + ", ".join(c["name"] for c in criteria) + "\n")
    for rank, item in enumerate(results, 1):
        sub = item["sub"]
        lines.append("## {}. {} - {} ({}/{} points)".format(
            rank, sub["team"], sub["project"], item["points"], item["max_points"]))
        lines.append("- Members: {}".format(sub["members"] or "not listed"))
        lines.append("- Link: {} ({})".format(sub["link"] or "none", sub["link_status"]))
        lines.append("- Scoring mode: {}".format(item["mode"]))
        for crit in criteria:
            score = item["scores"][crit["key"]]
            score_text = "{}/10".format(score) if score is not None else "not scored"
            lines.append("- {} ({}): {} - {}".format(
                crit["name"], score_text, item["evidence"][crit["key"]], crit["description"]))
        for note in item["notes"]:
            lines.append("- Note: {}".format(note))
        lines.append("")
    lines.append("Winners (top {}):".format(args.top))
    for rank, item in enumerate(results[: args.top], 1):
        lines.append("{}. {} - {}".format(rank, item["sub"]["team"], item["sub"]["project"]))
    lines.append("\nThe AI judge proposes. Human judges decide. Verify the top 3 by trying their demos.")
    with open(args.out, "w") as handle:
        handle.write("\n".join(lines))
    with open(os.path.splitext(args.out)[0] + ".json", "w") as handle:
        json.dump(results, handle, indent=2)
    print("\nReport written to {} (and JSON alongside it).".format(args.out))


def main():
    parser = argparse.ArgumentParser(description="Hackathon AI judge: rank public repos, pick the top 3.")
    parser.add_argument("repos", nargs="*", help="Repo URLs (or owner/repo).")
    parser.add_argument("--repos-file", help="File with one repo URL per line (# comments allowed).")
    parser.add_argument("--submissions", help="CSV export from a submission form (Google Forms). Runs submissions mode instead of repo mode.")
    parser.add_argument("--criteria", default="criteria.json", help="Criteria JSON file.")
    parser.add_argument("--llm", choices=["gemini", "groq", "none"], default="groq",
                        help="Scoring engine. groq needs GROQ_API_KEY, gemini needs GEMINI_API_KEY, else falls back.")
    parser.add_argument("--model", default=None,
                        help="Model name. Defaults: groq llama-3.3-70b-versatile, gemini gemini-2.0-flash. Override with GROQ_MODEL or GEMINI_MODEL.")
    parser.add_argument("--top", type=int, default=3, help="How many winners to pick.")
    parser.add_argument("--out", default="results.md", help="Markdown report path.")
    args = parser.parse_args()

    urls = list(args.repos)
    if args.repos_file:
        with open(args.repos_file) as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
    if not urls and not args.submissions:
        parser.error("Give at least one repo URL, a --repos-file, or --submissions.")

    with open(args.criteria) as handle:
        criteria = json.load(handle)["criteria"]
    total_weight = sum(c["weight"] for c in criteria)

    token = os.environ.get("GITHUB_TOKEN")
    if args.llm == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        model = args.model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    elif args.llm == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        model = args.model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    else:
        api_key, model = None, None
    if args.llm in ("gemini", "groq") and not api_key:
        key_name = "GROQ_API_KEY" if args.llm == "groq" else "GEMINI_API_KEY"
        print("No {} set, using rule based pre-screen for all entries.".format(key_name), file=sys.stderr)

    if args.submissions:
        run_submissions(args, criteria, args.llm, api_key, model)
        return

    results = []
    for url in urls:
        try:
            repo = fetch_repo(url, token)
        except ValueError as err:
            print("Skipping {}: {}".format(url, err), file=sys.stderr)
            continue
        if repo is None:
            print("Skipping {}: repo not found or not public.".format(url), file=sys.stderr)
            continue
        scores, evidence, notes, mode = score_repo(repo, criteria, args.llm, api_key, model)
        scored = [(c["weight"], scores[c["key"]]) for c in criteria if scores[c["key"]] is not None]
        points = round(sum(w * s for w, s in scored) / 10, 1) if scored else 0
        max_points = sum(w for w, _ in scored)
        ratio = (points / max_points) if max_points else 0
        results.append({
            "repo": repo, "scores": scores, "evidence": evidence,
            "notes": notes, "mode": mode, "points": points,
            "max_points": max_points, "ratio": ratio,
        })

    results.sort(key=lambda r: r["ratio"], reverse=True)

    print("\nRanked results (points out of measured criteria only):")
    for rank, item in enumerate(results, 1):
        mark = "  <-- TOP {}".format(rank) if rank <= args.top else ""
        print("{}. {} - {}/{}{}".format(rank, item["repo"]["name"], item["points"], item["max_points"], mark))
    print("\nWinners:")
    for rank, item in enumerate(results[: args.top], 1):
        print("  {}. {} ({})".format(rank, item["repo"]["name"], item["repo"]["url"]))

    lines = ["# Hackathon judging results\n"]
    lines.append("Scored against: " + ", ".join(c["name"] for c in criteria) + "\n")
    for rank, item in enumerate(results, 1):
        repo = item["repo"]
        lines.append("## {}. {} ({}/{} points)".format(rank, repo["name"], item["points"], item["max_points"]))
        lines.append("- Repo: {}".format(repo["url"]))
        lines.append("- Description: {}".format(repo["description"] or "none"))
        lines.append("- Scoring mode: {}".format(item["mode"]))
        for crit in criteria:
            score = item["scores"][crit["key"]]
            score_text = "{}/10".format(score) if score is not None else "not scored"
            lines.append("- {} ({}): {} - {}".format(
                crit["name"], score_text, item["evidence"][crit["key"]], crit["description"]))
        for note in item["notes"]:
            lines.append("- Note: {}".format(note))
        lines.append("")
    lines.append("Winners (top {}):".format(args.top))
    for rank, item in enumerate(results[: args.top], 1):
        lines.append("{}. {} - {}".format(rank, item["repo"]["name"], item["repo"]["url"]))
    lines.append("\nThe AI judge proposes. Human judges decide. Verify the top 3 by trying their demos.")
    with open(args.out, "w") as handle:
        handle.write("\n".join(lines))
    with open(os.path.splitext(args.out)[0] + ".json", "w") as handle:
        json.dump(results, handle, indent=2)
    print("\nReport written to {} (and JSON alongside it).".format(args.out))


if __name__ == "__main__":
    main()
