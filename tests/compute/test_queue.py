from finharness.compute.queue import ComputeJobStore, QueueFullError


def test_queue_leases_jobs_round_robin_across_users(tmp_path):
    store = ComputeJobStore(tmp_path / "compute_jobs.db")
    first = store.enqueue(user_id="u_a", conversation_id="c_a", kind="chart", payload_path="a-1.zip")
    store.enqueue(user_id="u_a", conversation_id="c_a", kind="chart", payload_path="a-2.zip")
    second = store.enqueue(user_id="u_b", conversation_id="c_b", kind="chart", payload_path="b-1.zip")

    leased_first = store.lease_next(worker_id="w_1", lease_seconds=30)
    leased_second = store.lease_next(worker_id="w_1", lease_seconds=30)

    assert leased_first is not None and leased_first.job_id == first.job_id
    assert leased_second is not None and leased_second.job_id == second.job_id


def test_queue_enforces_one_running_and_two_waiting_jobs_per_user(tmp_path):
    store = ComputeJobStore(tmp_path / "compute_jobs.db")
    running = store.enqueue(user_id="u_a", conversation_id="c", kind="chart", payload_path="1.zip")
    store.lease_next(worker_id="w_1", lease_seconds=30)
    store.mark_running(running.job_id, worker_id="w_1", lease_seconds=30)
    store.enqueue(user_id="u_a", conversation_id="c", kind="chart", payload_path="2.zip")
    store.enqueue(user_id="u_a", conversation_id="c", kind="chart", payload_path="3.zip")

    try:
        store.enqueue(user_id="u_a", conversation_id="c", kind="chart", payload_path="4.zip")
    except QueueFullError:
        pass
    else:
        raise AssertionError("third waiting job must be rejected")


def test_expired_lease_is_requeued_then_can_be_completed(tmp_path):
    store = ComputeJobStore(tmp_path / "compute_jobs.db")
    job = store.enqueue(user_id="u_a", conversation_id="c", kind="backtest", payload_path="job.zip")
    leased = store.lease_next(worker_id="lost", lease_seconds=0)

    assert leased is not None and leased.job_id == job.job_id
    recovered = store.lease_next(worker_id="live", lease_seconds=30)
    assert recovered is not None and recovered.job_id == job.job_id
    assert recovered.attempts == 2

    store.mark_running(job.job_id, worker_id="live", lease_seconds=30)
    store.succeed(job.job_id, worker_id="live", result_json='{"ok": true}')

    final = store.get(job.job_id, user_id="u_a")
    assert final is not None
    assert final.status == "succeeded"
    assert final.result_json == '{"ok": true}'
