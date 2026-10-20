import csv
cols = ["N","decaps","f_GHz","f_over_f1","rhoJ","Jacobi","dJacobi(2/3)","RB-GS","MG-V(Y)","GMRES50","GMRES50+ILU(1e-4)","BiCGStab","GMRES50+CSLexact","GMRES50+MG(Y)","GMRES50+MG(CSL)"]
print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
for N in (10, 20, 50, 100):
    for r in csv.DictReader(open(f"exp1_N{N}.csv")):
        if r["decaps"] != "none" and N in (10, 20): continue
        print("| " + " | ".join(r[c].replace("(r=", " (r=") for c in cols) + " |")
