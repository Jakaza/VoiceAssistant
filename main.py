from record import record_commands

if __name__ == "__main__":
    label = input("Enter label (e.g. move, sort, delete): ")
    for i in range(6):
        record_commands(label, i)


