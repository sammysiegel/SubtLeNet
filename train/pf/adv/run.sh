./train_conv.py > log 2>&1 & disown
echo "PID=$!"
tail -f log
