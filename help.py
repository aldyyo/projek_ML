help_glucose = ["Normal (tidak menderita diabetes): di bawah 117 mg/dL.",
                "Prediabetes: 117-137 mg/dL.",
                "Diabetes: 137 mg/dL atau lebih."
                ]

def HelpFunction(help_name):
    s = ''
    for i in help_name: 
        s += "- " + i + "\n"
    return s
