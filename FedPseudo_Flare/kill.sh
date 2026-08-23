PATTERN='job_fedpseudo\.py|client_module\.py|FedPseudo_Flare/client\.py|nvflare.*simulate_job'

mapfile -t PIDS < <(
    pgrep -u "$USER" -f "$PATTERN" || true
)

if ((${#PIDS[@]} == 0)); then
    echo "No FedPseudo_Flare processes are currently running."
else
    echo "Stopping these processes:"
    ps -o pid,ppid,stat,etime,args -p "$(IFS=,; echo "${PIDS[*]}")"

    kill -TERM "${PIDS[@]}"
    sleep 5

    # Force-stop only processes that ignored SIGTERM.
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force-killing PID $pid"
            kill -KILL "$pid"
        fi
    done
fi