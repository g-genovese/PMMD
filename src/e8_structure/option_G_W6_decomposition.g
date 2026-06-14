# option_G_W6_decomposition.g
# ============================================================================
# Verifies, against the GAP character-table library, the W(E_6)-module
# decomposition of the W(E_8) irreducible V_112
# (paper: Prop Z3-A2-Coxeter-identification, Cor V112-W6-decomposition).
#
#   (A)/(B) chi_{V112} on the four order-3 classes; the A_2-Coxeter class
#           (centraliser |W(A_2)xW(E_6)| = 311040, size 2240) gives the Koide
#           Z_3-eigenspaces (58,27,27); the other three (sizes 4480, 89600,
#           268800) give (40,36,36);
#   (C)     V_112|_{W(E_6)} via PossibleClassFusions decomposes with W(E_6)-
#           irrep dimension multiset {1:4, 6:3, 20:3, 30:1} = 112, i.e.
#           58 (+) 2*27 with 27 = 1+6+20 and 58 = 2*1+6+20+30.
#
# Run:  gap -q -b option_G_W6_decomposition.g
# ============================================================================
LoadPackage("ctbllib");;
ct8 := CharacterTable("W(E8)");;
ct6 := CharacterTable("W(E6)");;
irr8 := Irr(ct8);;  irr6 := Irr(ct6);;
ords  := OrdersClassRepresentatives(ct8);;
sizes := SizesConjugacyClasses(ct8);;
cent  := SizesCentralizers(ct8);;

v112_idx := Filtered([1..Length(irr8)], i -> irr8[i][1] = 112);;
Print("112-dim W(E8) irreps at positions: ", v112_idx, "\n");
chi := irr8[v112_idx[1]];;

o3 := Filtered([1..Length(ords)], i -> ords[i] = 3);;
Print("\n(A)/(B) order-3 classes  [class: size, centraliser, chi_112 -> (a,b,b)]:\n");
for j in o3 do
  a := (chi[1] + 2*chi[j]) / 3;
  b := (chi[1] - chi[j]) / 3;
  Print("  class ", j, ": size=", sizes[j], "  cent=", cent[j],
        "  chi_112=", chi[j], "  -> (", a, ",", b, ",", b, ")\n");
od;
a2 := Filtered(o3, j -> cent[j] = 311040);;
Print("A2-Coxeter class (cent=311040): ", a2, "  size(s)=", List(a2, j -> sizes[j]), "\n");

Print("\n(C) restriction V112|_{W(E6)} via PossibleClassFusions:\n");
fus := PossibleClassFusions(ct6, ct8);;
Print("  #possible fusions: ", Length(fus), "\n");
seen := [];;
for f in fus do
  restricted := chi{f};;
  dec := [];;
  for x in irr6 do Add(dec, ScalarProduct(ct6, restricted, x)); od;
  comps := [];;
  for i in [1..Length(irr6)] do
    if dec[i] > 0 then Add(comps, [irr6[i][1], dec[i]]); fi;
  od;
  Sort(comps);
  if not comps in seen then
    Add(seen, comps);
    Print("  -> components [dim,mult]: ", comps,
          "   total=", Sum(comps, c -> c[1]*c[2]), "\n");
  fi;
od;
QUIT;
