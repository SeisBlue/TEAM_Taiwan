import random
import threading
import time
from collections import deque


# 模擬接收的波形數據（例如：每秒產生一個隨機數字）
def data_receiver(buffer, buffer_lock, stop_event):
    while not stop_event.is_set():
        data = random.random()  # 模擬1秒的波形數據
        with buffer_lock:
            buffer.append((time.time(), data))  # 儲存時間戳和數據
            print(f"接收波形數據: {data:.4f}")
        time.sleep(1)  # 模擬1秒接收一次


# 模擬 P 波檢測（隨機在10到30秒內觸發一次P波）
def p_wave_detector(trigger_event, stop_event):
    while not stop_event.is_set():
        # 隨機等待10到30秒後觸發P波
        wait_time = random.randint(3, 15)
        print(f"P 波偵測器將在 {wait_time} 秒後觸發 P 波事件")
        time.sleep(wait_time)
        print("偵測到 P 波!")
        trigger_event.set()  # 設置觸發事件
        # 等待處理完成後重置事件
        trigger_event.clear()


# 觸發後的處理函數
def trigger_handler(buffer, buffer_lock, trigger_event, stop_event):
    while not stop_event.is_set():
        # 等待 P 波觸發
        triggered = trigger_event.wait(timeout=1)
        if triggered:
            with buffer_lock:
                current_time = time.time()
                # 提取 P 波前的 5 秒數據
                pre_trigger_data = [
                    data for (ts, data) in buffer if ts >= current_time - 5
                ]
            print("P 波被偵測到，處理前 5 秒波形數據:")
            print(pre_trigger_data)

            # 收集 P 波當下及後的 3 秒數據
            post_trigger_data = []
            print("開始收集 P 波後的 3 秒數據...")
            for _ in range(3):
                if stop_event.is_set():
                    break
                time.sleep(1)
                with buffer_lock:
                    if buffer:
                        ts, data = buffer[-1]  # 取得最新的數據
                        post_trigger_data.append(data)
                        print(f"收集到後續波形數據: {data:.4f}")
            print("P 波當下及後的 3 秒波形數據:")
            print(post_trigger_data)

            # 設置一個事件來開始每隔2秒輸出數據
            periodic_trigger = threading.Event()
            periodic_stop_event = threading.Event()
            periodic_thread = threading.Thread(
                target=periodic_output,
                args=(buffer, buffer_lock, periodic_trigger, periodic_stop_event),
            )
            periodic_thread.start()

            # 設置定時器，每2秒觸發一次輸出
            def periodic_timer():
                while not stop_event.is_set():
                    time.sleep(2)
                    periodic_trigger.set()

            timer_thread = threading.Thread(target=periodic_timer)
            timer_thread.start()

        time.sleep(0.1)


# 每隔2秒輸出一次數據的處理函數
def periodic_output(buffer, buffer_lock, periodic_trigger, periodic_stop_event):
    while not periodic_stop_event.is_set():
        triggered = periodic_trigger.wait(timeout=1)
        if triggered:
            with buffer_lock:
                current_time = time.time()
                # 例如，輸出最近的2秒數據
                recent_data = [data for (ts, data) in buffer if ts >= current_time - 2]
            print("每隔2秒輸出一次最新的2秒波形數據:")
            print(recent_data)
            periodic_trigger.clear()
    print("周期性輸出線程已終止。")


def main():
    # 緩衝區初始化（最多存儲10個1秒的數據，至少存儲5秒前的數據）
    buffer = deque(maxlen=100)  # 增大緩衝區以確保有足夠的前置數據
    buffer_lock = threading.Lock()

    # 事件初始化
    trigger_event = threading.Event()
    stop_event = threading.Event()

    # 創建線程
    receiver_thread = threading.Thread(
        target=data_receiver, args=(buffer, buffer_lock, stop_event)
    )
    detector_thread = threading.Thread(
        target=p_wave_detector, args=(trigger_event, stop_event)
    )
    handler_thread = threading.Thread(
        target=trigger_handler, args=(buffer, buffer_lock, trigger_event, stop_event)
    )

    # 啟動線程
    receiver_thread.start()
    detector_thread.start()
    handler_thread.start()

    try:
        # 主線程等待，直到用戶中斷
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("接收到中斷信號，正在關閉...")
        stop_event.set()

    # 等待所有線程結束
    receiver_thread.join()
    detector_thread.join()
    handler_thread.join()
    print("程序已結束。")


if __name__ == "__main__":
    main()
