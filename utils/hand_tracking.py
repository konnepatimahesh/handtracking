def get_gesture_prediction(hand_landmarks):
    """
    Predicts a gesture: 'Start', 'Stop', 'Run', 'Peace', 'ThumbsUp'
    using MediaPipe hand landmarks.
    """

    # Tips and lower joints
    finger_tips = [8, 12, 16, 20]     # Index, Middle, Ring, Pinky
    finger_pips = [6, 10, 14, 18]

    # Count extended fingers
    extended = []
    for tip, pip in zip(finger_tips, finger_pips):
        is_extended = hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
        extended.append(is_extended)

    index, middle, ring, pinky = extended
    extended_count = extended.count(True)

    # Thumb logic (based on x for horizontal, y for vertical hand)
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_extended = thumb_tip.x < thumb_ip.x  # for right hand

    # Rule-based gesture classification
    if extended_count == 0 and not thumb_extended:
        return "Start"
    elif extended_count == 4 and thumb_extended:
        return "Stop"
    elif index and not middle and not ring and not pinky and thumb_extended:
        return "Run"
    elif index and middle and not ring and not pinky:
        return "Peace"
    elif not index and not middle and not ring and not pinky and thumb_extended:
        return "ThumbsUp"
    else:
        return "Unknown"
