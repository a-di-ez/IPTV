#!/usr/bin/python3
"""Report which channels in `lists/` are no longer reachable.

The playlist checker records the URLs it could not open, but a bare URL does not say
which channel or which list it came from, and one failed attempt does not mean a
channel is gone. Streams time out, rate limit, move between CDNs, and some are only
served inside their own country.

This script maps every URL back to its channel and list, probes each one several
times, and separates the cases that look alike from the outside:

  dead         every attempt said "not found", so the stream is really gone
  stubbed      the server served a playlist, but the only thing in it is a
               placeholder clip - an error card standing in for the channel
  disputed     looked dead over HTTP, but ffprobe found a real audio/video stream
               behind it - needs a human, not a bot, to decide
  blocked      the server answered but refused us, which is what a channel served
               only in its own country looks like from anywhere else
  unreachable  every attempt timed out or failed to connect
  flaky        answered at least once, so it is up but unreliable from here
  alive        answered every time

Only `dead` and `stubbed` are safe to act on without a second opinion. `blocked` in
particular must not be treated as a fault: the lists mark geo-blocked channels
deliberately.

Both playlist formats used by the lists are understood, HLS (.m3u8) and MPEG-DASH
(.mpd), so a DASH stream is not mistaken for a broken one. Neither check is airtight
on its own: a manifest that starts with `#EXTM3U`/`<MPD` is not proof the media
behind it plays, and a payload ffprobe can decode is not proof it is a live channel
rather than a stray media fragment sitting at a familiar-looking URL - one such
fragment is what first prompted the --confirm-dead option below. Treat `dead` as
"nothing here answered like a stream, twice, two different ways" rather than proof.

`stubbed` exists because those two checks share a blind spot, and 13 channels sat in
the Ukrainian list for months because of it. A CDN whose token has expired need not
answer 403: cdnua05.hls.tv answered 200 with a manifest that parsed cleanly and whose
one segment was `/stub_55x/token.ts`, a card reading "недоступний токен". The header
check saw `#EXTM3U` and said alive; ffprobe, asked for a second opinion, decoded the
card and reported healthy 1280x720 video - because an error card *is* real video. No
check that stops at the manifest, or that only asks "does something decode", can tell
those apart. Reading the segment names inside the manifest can, so HLS playlists are
now opened one level deeper.

With --confirm-dead, every channel that looks `dead` over HTTP gets a second,
independent opinion from `ffprobe` (part of ffmpeg) before being reported as dead:
ffprobe actually tries to decode audio/video from the URL, which catches streams a
header check alone cannot judge either way. This second opinion is one-directional -
it can only pull a channel *out* of `dead` into `disputed`, never push a channel that
looks fine over HTTP into a worse state - because a slow or unusual server can make
ffprobe time out on a channel that plays fine elsewhere (this happened while writing
this script: a known-good DASH channel needed longer than any reasonable per-channel
budget for ffprobe to open), and a timeout there must not be read as confirmation of
anything. It is off by default because it materially changes the runtime: ffprobe
does real network I/O per candidate, on top of the HTTP probes already spent finding
that candidate, and only makes sense where that time is available (a weekly run),
not where it is not (a PR check blocking on a handful of changed links).

Usage:
    ./check_channels.py                          # every list
    ./check_channels.py greece italy             # only those lists
    ./check_channels.py --attempts 5 greece      # more attempts per channel
    ./check_channels.py --confirm-dead greece    # + ffprobe second opinion on dead ones
"""

import argparse
import http.client
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

LISTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lists")

# one bundle for the knobs every probe/list/run function otherwise had to repeat
ProbeSettings = namedtuple(
    "ProbeSettings", ["attempts", "timeout", "pause", "workers", "confirm_dead"],
)

# `| 1 | Channel name | [>](url) | ...` is the row format used by every list
ROW = re.compile(r"^\|[^|]*\|([^|]+)\|[^|]*\[>\]\((https?://[^)\s]+)\)")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)

OK = "ok"
GONE = "gone"
STUB = "stub"
REFUSED = "refused"
UNREACHABLE = "unreachable"

ALIVE = "alive"
DEAD = "dead"
STUBBED = "stubbed"
DISPUTED = "disputed"
BLOCKED = "blocked"
UNREACHED = "unreachable"
FLAKY = "flaky"

# a segment named after a failure rather than after a time or a sequence number.
# Matched against the segment's path only, never its query: a real segment URL
# routinely carries `?token=...` and must not be read as a placeholder.
PLACEHOLDER = re.compile(r"stub|placeholder|no[_-]?signal|offline|unavailable|blocked|error")

# a channel served only in its own country answers, then refuses us
REFUSING_CODES = (401, 402, 403, 451)
GONE_CODES = (404, 410)

# ffprobe gets much longer than an HTTP probe: it has to open a connection, read
# enough of the stream to find a decodable frame, and do that over whatever the
# channel's own CDN feels like doing today, not just get a response header back
FFPROBE_TIMEOUT = 25


def read_channels(name):
    """Return the `(channel, url)` pairs of the list called `name`."""
    path = os.path.join(LISTS_DIR, name + ".md")
    channels = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            match = ROW.match(line.strip())
            if match:
                channels.append((match.group(1).strip(), match.group(2)))
    return channels


def looks_like_a_playlist(head):
    """Return True if `head` is the start of an HLS or a DASH playlist."""
    text = head.lstrip()
    if text.startswith("#EXTM3U"):
        return True
    # DASH manifests are XML, and may carry a declaration, a comment or neither
    return "<MPD" in text[:2000]


def read_head(url, timeout, limit=8000):
    """Open `url` once and return `(outcome, the first bytes of the body)`.

    The outcome is OK whenever the server gave us a body at all - "something
    answered", not yet "this is a playlist" - and GONE/REFUSED/UNREACHABLE when it
    did not, in which case the body is empty.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    # a stream whose certificate does not verify is still a working stream
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            return OK, response.read(limit).decode("utf-8", "ignore")
    except urllib.error.HTTPError as error:
        if error.code in GONE_CODES:
            return GONE, ""
        if error.code in REFUSING_CODES:
            return REFUSED, ""
        return UNREACHABLE, ""
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
        # http.client.HTTPException covers a server that started a chunked
        # response and then hung up mid-chunk (IncompleteRead) and similar
        # low-level protocol violations - a broken connection, not a bad URL
        return UNREACHABLE, ""


def playlist_entries(body):
    """The non-comment lines of an HLS playlist: its variants, or its segments."""
    return [line.strip() for line in body.splitlines() if line.strip() and not line.startswith("#")]


def serves_a_placeholder(url, body, timeout):
    """Return True if the HLS playlist at `url` offers nothing but a placeholder.

    Two signatures count, and the `/stub_55x/token.ts` card described in the module
    docstring trips both: a segment path named after a failure, and a playlist whose
    segments are all the same file, which no live channel produces.

    A master playlist lists variants rather than segments, so the placeholder sits
    one level down and this follows the first variant to find it. When that second
    request fails the answer is False: not being able to look is not evidence of a
    stub, and this must never be the reason a working channel is called broken.
    """
    entries = playlist_entries(body)
    if not entries:
        return False
    variant = next((entry for entry in entries if ".m3u8" in entry or ".m3u" in entry), None)
    if variant is not None:
        outcome, body = read_head(urllib.parse.urljoin(url, variant), timeout)
        if outcome != OK or not body.lstrip().startswith("#EXTM3U"):
            return False
        entries = playlist_entries(body)
        if not entries:
            return False
    paths = [urllib.parse.urlsplit(entry).path.lower() for entry in entries]
    if any(PLACEHOLDER.search(path) for path in paths):
        return True
    return len(entries) > 1 and len(set(entries)) == 1


def probe(url, timeout):
    """Open `url` once and report what the server did."""
    outcome, head = read_head(url, timeout)
    if outcome != OK:
        return outcome
    if not looks_like_a_playlist(head):
        return GONE
    # only HLS is opened a level deeper; a DASH manifest describes its segments
    # in the XML itself rather than pointing at a list of them
    if head.lstrip().startswith("#EXTM3U") and serves_a_placeholder(url, head, timeout):
        return STUB
    return OK


def ffprobe_finds_a_stream(url):
    """Ask ffprobe whether it can decode real audio/video from `url`.

    Returns True/False for a definite answer, or None if ffprobe itself could not
    be asked (missing binary, timed out, or errored) - None must never be treated
    as "no stream found", only as "no second opinion available this time".
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-user_agent", USER_AGENT,
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                url,
            ],
            capture_output=True, text=True, timeout=FFPROBE_TIMEOUT, check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    except OSError:
        return None
    if result.returncode != 0:
        return False
    return any(kind in result.stdout for kind in ("video", "audio"))


def classify(outcomes):
    """Turn the outcomes of the attempts on one channel into a single state."""
    if all(outcome == OK for outcome in outcomes):
        return ALIVE
    if any(outcome == OK for outcome in outcomes):
        return FLAKY
    # a stub is the server stating the channel will not play, which outranks a
    # refusal or a timeout: those only say we could not get to it this time
    if any(outcome == STUB for outcome in outcomes):
        return STUBBED
    if any(outcome == REFUSED for outcome in outcomes):
        return BLOCKED
    if all(outcome == GONE for outcome in outcomes):
        return DEAD
    return UNREACHED


def check(channel, settings):
    """Probe one channel `settings.attempts` times and return its state.

    If the channel looks dead and `settings.confirm_dead` is set, give it one more
    chance through ffprobe before reporting it as dead - see the module docstring
    for why this can only pull a verdict out of `dead`, never push one into it.

    A `stubbed` channel is pointedly not offered that second chance. ffprobe decodes
    a placeholder card as happily as a channel, so asking it here would do nothing
    but overturn the one check that saw through the stub.
    """
    name, url = channel
    outcomes = []
    for attempt in range(settings.attempts):
        outcomes.append(probe(url, settings.timeout))
        if attempt + 1 < settings.attempts:
            time.sleep(settings.pause)
    state = classify(outcomes)
    if state == DEAD and settings.confirm_dead:
        if ffprobe_finds_a_stream(url):
            state = DISPUTED
    return name, url, state, outcomes


def check_all(names, settings):
    """Probe every channel of every named list and return `{list_name: [results]}`.

    All channels of all lists share one pool of `settings.workers`, so a run across
    many lists is not slower per list than a run of one - a list with two channels
    does not pay the same wall-clock floor as a list with two hundred just because
    it was handed its own pool that then sits mostly idle.
    """
    channels_by_list = {
        name: [c for c in read_channels(name) if not c[1].startswith("https://www.youtube.com")]
        for name in names
    }

    def run(item):
        list_name, channel = item
        return list_name, check(channel, settings)

    jobs = [(name, channel) for name, channels in channels_by_list.items() for channel in channels]
    results_by_list = {name: [] for name in names}
    with ThreadPoolExecutor(max_workers=settings.workers) as pool:
        for list_name, result in pool.map(run, jobs):
            results_by_list[list_name].append(result)
    return results_by_list


def report(name, results, attempts):
    """Print the results of one list and return how many channels are provably broken.

    Provably broken is `dead` plus `stubbed`: both are the server telling us the
    channel is not there, unlike a refusal or a timeout, which only describe the
    trip. Those two are what the exit code is for.
    """
    states = {}
    for _, _, state, _ in results:
        states[state] = states.get(state, 0) + 1
    summary = ", ".join(f"{states[s]} {s}" for s in sorted(states))
    print(f"{name}: {len(results)} checked, {summary}")
    for channel, url, state, outcomes in results:
        if state == ALIVE:
            continue
        detail = "/".join(outcomes) if state != FLAKY else f"{outcomes.count(OK)}/{attempts} ok"
        print(f"  {state:12} {channel} [{detail}] -> {url}")
    return states.get(DEAD, 0) + states.get(STUBBED, 0)


def as_records(list_name, results, checked_at, confirm_dead):
    """Turn one list's results into flat dicts, one per channel, for --json output.

    `confirm_dead` is recorded on every row so a later reader can tell a `dead`
    verdict that already survived an ffprobe second opinion from one that has not
    been asked yet - the two are not the same strength of evidence.
    """
    return [
        {
            "checked_at": checked_at,
            "list": list_name,
            "channel": channel,
            "url": url,
            "state": state,
            "outcomes": outcomes,
            "confirm_dead": confirm_dead,
        }
        for channel, url, state, outcomes in results
    ]


def parse_args():
    """Parse the command line into an `argparse.Namespace`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("lists", nargs="*", help="lists to check, without the .md suffix")
    parser.add_argument("--attempts", type=int, default=3, help="probes per channel")
    parser.add_argument("--timeout", type=int, default=12, help="seconds per probe")
    parser.add_argument("--pause", type=int, default=5, help="seconds between probes")
    parser.add_argument("--workers", type=int, default=8, help="channels probed at once")
    parser.add_argument(
        "--confirm-dead", action="store_true",
        help="give ffprobe a second opinion on channels that look dead (slower; needs ffmpeg)",
    )
    parser.add_argument(
        "--json", metavar="PATH",
        help="also write every channel's result as one JSON record per line to PATH, for "
             "building a history across runs (state, outcomes, timestamp, list, channel, url)",
    )
    return parser.parse_args()


def write_json_records(handle, names, results_by_list, checked_at, confirm_dead):
    """Write every list's results to `handle` as one JSON record per channel per line."""
    for name in names:
        for record in as_records(name, results_by_list[name], checked_at, confirm_dead):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    """Check the lists named on the command line, or every list."""
    args = parse_args()

    if args.confirm_dead and shutil.which("ffprobe") is None:
        print(
            "--confirm-dead needs ffprobe (part of ffmpeg) on PATH; proceeding without it.",
            file=sys.stderr,
        )
        args.confirm_dead = False

    all_lists = (f[:-3] for f in os.listdir(LISTS_DIR) if f.endswith(".md"))
    names = args.lists or sorted(all_lists)
    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    settings = ProbeSettings(
        args.attempts, args.timeout, args.pause, args.workers, args.confirm_dead,
    )

    results_by_list = check_all(names, settings)

    broken_total = sum(report(name, results_by_list[name], args.attempts) for name in names)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            write_json_records(handle, names, results_by_list, checked_at, args.confirm_dead)

    return 1 if broken_total else 0


if __name__ == "__main__":
    sys.exit(main())
