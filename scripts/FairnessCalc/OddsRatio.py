def calculate_odds_ratio(A, B, C, D):
    try:
        OR = (A * D) / (B * C)
        return OR
    except ZeroDivisionError:
        return "Undefined (division by zero)"
    
A = int(input("Enter the number of cases with exposure: "))
B = int(input("Enter the number of controls with exposure: "))
C = int(input("Enter the number of cases without exposure: "))
D = int(input("Enter the number of controls without exposure: "))

odds_ratio = calculate_odds_ratio(A, B, C, D)
print(f"Odds Ratio: {odds_ratio}")
