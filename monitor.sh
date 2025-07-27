# The username to monitor
TARGET_USER="stella"

# The PIDs to kill directly
PROCESS_NAME1="681303"
PROCESS_NAME2="681052"

echo "Monitoring logins for user: $TARGET_USER..."
echo "process names: $PROCESS_NAME1, $PROCESS_NAME2"

gpus_to_kill="0 1"

while true; do
    # Check if the target user is logged in
    if who | awk '{print $1}' | grep -q "^$TARGET_USER$"; then
        echo "$TARGET_USER has logged in. Killing all GPU processes run by user dh6dh..."
        # Get all PIDs of GPU processes run by dh6dh
        dh6dh_pids=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | awk -F',' '{print $1}' | xargs -r ps -u dh6dh -o pid= | awk '{print $1}')
        for PID in $dh6dh_pids; do
            if [ -n "$PID" ]; then
                echo "Killing GPU process PID $PID run by dh6dh"
                kill -9 "$PID"
            fi
        done
    fi
    # Wait for a short interval before checking again
    sleep 2
done


while true; do
    echo "!!!StopStopStopStopStopStopStopStopStopStopStopStopStopStopStopStopStopStopStopStopStopStop!!!"
    sleep 0.1
done