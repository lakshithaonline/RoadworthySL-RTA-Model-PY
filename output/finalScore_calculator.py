import random
import tkinter as tk
from tkinter import messagebox

# Dataset
data = {
    'Tyres': 0,
    'Brakes': 0,
    'Suspension': 0,
    'Body and Chassis': 0,
    'Lights': 0,
    'Glazing': 0,
    'Wipers': 0,
    'Doors': 0,
    'Seat Belts': 0,
    'Airbags': 0,
    'Speedometer': 0,
    'Exhaust System': 0,
    'Fuel System': 0
}

# Weights and Criticality
WEIGHTS = {
    'High': 10,
    'Medium': 6,
    'Low': 4
}

PARAMETER_CRITICALITY = {
    'Tyres': 'High',
    'Brakes': 'High',
    'Suspension': 'Medium',
    'Body and Chassis': 'Medium',
    'Lights': 'High',
    'Glazing': 'Medium',
    'Wipers': 'Low',
    'Doors': 'Low',
    'Seat Belts': 'Medium',
    'Airbags': 'High',
    'Speedometer': 'Low',
    'Exhaust System': 'Medium',
    'Fuel System': 'High'
}


def calculate_final_score():
    try:
        for param in data:
            if param != 'Final Score':
                data[param] = int(entries[param].get())

        parameter_scores = [(data[param], PARAMETER_CRITICALITY[param]) for param in data if param != 'Final Score']
        weighted_scores = [score * WEIGHTS[critical_level] for score, critical_level in parameter_scores]
        total_weighted_score = sum(weighted_scores)
        max_possible_weighted_score = sum([10 * WEIGHTS[critical_level] for _, critical_level in parameter_scores])
        final_percentage_score = (total_weighted_score / max_possible_weighted_score) * 100

        # Update the final score label
        score_label.config(text=f"Final Score: {final_percentage_score:.2f}")

        # Update the final score in data
        data['Final Score'] = round(final_percentage_score, 2)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid integers for all parameters.")


def generate_random_scores():
    for param in data:
        if param != 'Final Score':
            random_score = random.randint(1, 10)
            entries[param].delete(0, tk.END)
            entries[param].insert(0, random_score)


def copy_to_clipboard():
    try:
        result_str = "random_values = {\n"
        for param in data:
            result_str += f"    '{param}': {int(entries[param].get()) if param != 'Final Score' else data['Final Score']},\n"
        result_str = result_str.rstrip(',\n') + "\n}"

        root.clipboard_clear()
        root.clipboard_append(result_str)
        messagebox.showinfo("Copy to Clipboard", "Data copied to clipboard successfully!")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid integers for all parameters.")


# Create main window
root = tk.Tk()
root.title("Vehicle Inspection Score Calculator")
root.geometry("720x680")
root.resizable(True, True)
root.eval('tk::PlaceWindow . center')

# Set theme colors
bg_color = "#f5f5f5"
entry_bg_color = "#ffffff"
btn_color = "#e0e0e0"

root.configure(bg=bg_color)

# Create a canvas and scrollbar
canvas = tk.Canvas(root, bg=bg_color)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg=bg_color)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Pack canvas and scrollbar
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Create and place labels and entries
entries = {}
row = 0
for param in data:
    if param != 'Final Score':
        label = tk.Label(scrollable_frame, text=param, bg=bg_color, font=('Helvetica', 12))
        label.grid(row=row, column=0, padx=10, pady=5, sticky='w')
        entry = tk.Entry(scrollable_frame, bg=entry_bg_color, font=('Helvetica', 12))
        entry.grid(row=row, column=1, padx=10, pady=5, sticky='w')
        entries[param] = entry
        row += 1

# Final score label
score_label = tk.Label(scrollable_frame, text="Final Score: ", bg=bg_color, font=('Helvetica', 14))
score_label.grid(row=row, column=0, padx=10, pady=10, sticky='w')

# Calculate button
calculate_btn = tk.Button(scrollable_frame, text="Calculate Final Score", bg=btn_color, font=('Helvetica', 12),
                          command=calculate_final_score)
calculate_btn.grid(row=row, column=1, pady=20, sticky='e')

# Generate random scores button
generate_random_btn = tk.Button(scrollable_frame, text="Generate Random Scores", bg=btn_color, font=('Helvetica', 12),
                                command=generate_random_scores)
generate_random_btn.grid(row=row + 1, column=0, pady=10, padx=10, sticky='w')

# Copy to clipboard button
copy_btn = tk.Button(scrollable_frame, text="Copy to Clipboard", bg=btn_color, font=('Helvetica', 12),
                     command=copy_to_clipboard)
copy_btn.grid(row=row + 2, column=1, pady=10, sticky='e')

# Center window
root.update_idletasks()
width = root.winfo_width()
height = root.winfo_height()
x = (root.winfo_screenwidth() // 2) - (width // 2)
y = (root.winfo_screenheight() // 2) - (height // 2)
root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

root.mainloop()
