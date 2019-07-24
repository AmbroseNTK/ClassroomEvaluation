
LOG_FRAMING = 0
LOG_BEHAVIOR = 1
LOG_FACIAL = 2
LOG_MOVEMENT = 3

log_files = ["frames", "behaviors", "facial", "movement"]


def write_log(log_type_id, sess_id, progress, total):
    file = open("result/" + sess_id + "/logs/" +
                log_files[log_type_id] + ".log", "w")
    file.write(total)
    file.write(progress)
    file.close()
