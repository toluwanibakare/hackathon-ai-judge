# Hackathon AI judge

Ranks public GitHub repos against a set of judging criteria and picks the top 3.

## How to run

```
python3 judge.py --repos repos.txt --criteria criteria.json --out results.md
```

Or pass repos directly:

```
python3 judge.py https://github.com/owner/repo https://github.com/owner/other
```

`repos.txt` holds one repo URL per line. Lines starting with `#` are ignored.

## Scoring

Two modes:

- **LLM mode (full judging).** Set a free Groq API key and the judge scores
  every criterion with evidence quoted from the README:
  ```
  export GROQ_API_KEY=your_key_here
  python3 judge.py --repos repos.txt --llm groq
  ```
  Get a free key from the Groq console (console.groq.com). The default
  model is llama-3.3-70b-versatile; change it with GROQ_MODEL or --model.
  Gemini is still available with `--llm gemini` and GEMINI_API_KEY.
  If the key is missing or the call fails, the judge falls back to
  pre-screen mode for that repo.
- **Pre-screen mode (no key needed).** Scores only what can be measured:
  README quality and demo evidence. Usefulness and creativity are marked
  "needs a human judge". Totals are shown out of measured points only
  (for example 48/50), never out of 100, so a pre-screen score is never
  mistaken for a full judgement. Use it to shortlist, then let human
  judges decide.

Public repos need no GitHub key. If you hit the rate limit (60 requests an
hour without a key), set `GITHUB_TOKEN` to raise it.

## Criteria

Edit `criteria.json` to change criteria and weights. The default set is the
fresher AI build day set: working demo (30), usefulness for a fresher (30),
creativity (20), presentation and README (20).

## At the event

1. Collect one repo URL per team into `repos.txt`.
2. Run the judge to get a ranked shortlist.
3. Human judges try the demos of the shortlisted projects live and confirm
   the top 3. The AI judge proposes, humans decide.
4. Announce from the report, not from the raw scores alone.

## Honest limits

- A great README is not proof a project works. Always try the demo.
- A weak README does not mean a weak project. Check the demo before
  eliminating anyone on docs alone.
- Private repos cannot be judged. Teams must make their repo public or
  submit a public demo link.
- The judge reads at most 12000 characters of each README.
