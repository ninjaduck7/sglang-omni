# Local qualification protocol

The endpoints and fixtures below are local. Preserve each run under a fresh directory.
Do not change the resident limit, reuse another benchmark's private cache or mix
historical serving revisions. Compare results to the criteria below without
reclassifying client disconnects or a terminal HTTP response as server cancellation.

| Path | Criterion set before the soak |
| --- | --- |
| Historical short-form quality | Account for all IDs, repeat cells, reference units, errors and edits. Report observed ranges. No new retrospective H100 parity threshold. |
| Mixed stability | Four closed-loop cells of at least 450 s with the same server identity. Count all ordinary completions and all probe outcomes. Zero unexpected timeouts/5xx on valid traffic. Score fixed control transcripts separately. |
| Error isolation | Empty file, malformed audio and invalid parameters return 4xx; new independent fixed transcript and health succeed. 5xx fails the rejection criterion even if recovery succeeds. |
| Disconnect | Start from idle, observe exactly one pending request with no competing traffic. Disconnect it, sample pending state, then issue a new independent request. Claim backend cancellation only with request-specific abort evidence. |
| Streaming | Request a stream; require incremental text events and a terminal result for streaming/first-token timing. Terminal JSON alone does not qualify. |
| Boundary | Place known speech immediately before 30 s and wholly after 30 s with an identical prefix. Check for its words. Probe a much longer input or oversized context; distinguish HTTP acceptance, feature truncation and an actual decoder-context rejection. |
| Active shutdown | From idle start a request, observe it pending, signal the real server PID. Record the client outcome, port, all observed descendants and surviving service GPU PIDs. Rebind same port and recheck transcript. Distinguish guard group cleanup from server-only graceful shutdown. |
| Startup and memory | Disk-resident verified weights. First run with empty private JIT paths, then same paths with a new process. Record spawn-to-health and spawn-to-correct-output. Record board memory separately from descendants and other compute PIDs; warm and cooldown must use the same process. A single cooldown is not proof of leak freedom. |
| Resource guard | One service; existing guard requires 14 GiB free to launch and samples for a 4 GiB floor. Require 8 GiB free temporary headroom before launch. Stop only own processes. |

Raw downloads, private compiler cache and model files are transient. The local evidence directory
contains per-request records, probe outcomes, sampled resources and package/model manifests.
The existing full-corpus sweep is not repeated.
