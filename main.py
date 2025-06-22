from record import record_commands

if __name__ == "__main__":
    label = input("Enter full phrase (e.g. 'open chrome'): ").strip().lower().replace(" ", "_")
    for i in range(6):
        record_commands(label, i)


