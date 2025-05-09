def seconds_to_dhms_str(seconds):
    """Convert seconds to days, hours, minutes, seconds string"""
    days = int(seconds // (3600 * 24))
    hours = int((seconds // 3600) % 24)
    minutes = int((seconds // 60) % 60)
    seconds = int(seconds % 60)

    if days < 1:
        if hours < 1:
            if minutes < 1:
                return f"{seconds}s"
            return f"{minutes}m {seconds}s"
        return f"{hours}h {minutes}m {seconds}s"

    return f"""{days}d {hours}h {minutes}m {seconds}s"""
