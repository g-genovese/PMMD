# w_e8_character_analysis.gap
# ============================================================================
# W(E_8) character analysis against the GAP character-table library
# (paper: Thm Rperm-decomposition and the appendix GAP-verified items).
#
#   (1) the permutation character R_perm of W(E_8) on the 240 E_8 roots
#       (= trivial char of the point stabiliser W(E_7) induced to W(E_8))
#       decomposes as 1 (+) 8 (+) 35 (+) 84 (+) 112, with <R_perm,R_perm> = 5;
#   (2) the character of V_112 on the four order-3 classes ([31,4,4,4]);
#   (3) under the A_2-Coxeter order-3 element, 72 of the 240 roots are fixed
#       (the E_6 root system); the remaining 168 form 56 orbits of size 3
#       (the matter content).
#
# NOTE: an earlier draft built the table from scratch via
# SimpleLieAlgebra("E",8) -> WeylGroup -> CharacterTable, infeasible for the
# 696,729,600-element W(E_8). This version uses the library tables
# CharacterTable("W(E8)") / CharacterTable("W(E7)") and runs in seconds.
# The W(E_6)-module decomposition of V_112 is in option_G_W6_decomposition.g.
#
# Run:  gap -q -b w_e8_character_analysis.gap
# ============================================================================
LoadPackage("ctbllib");;
ct8 := CharacterTable("W(E8)");;
ct7 := CharacterTable("W(E7)");;
Print("|W(E8)| = ", Size(ct8), ",  conjugacy classes = ", NrConjugacyClasses(ct8), "\n");
Print("point stabiliser |W(E7)| = ", Size(ct7),
      "  (= |W(E8)|/240 = ", Size(ct8)/240, ")\n");
irr8  := Irr(ct8);;
ords  := OrdersClassRepresentatives(ct8);;
cent  := SizesCentralizers(ct8);;
sizes := SizesConjugacyClasses(ct8);;

# ---- (1) R_perm on the 240 roots = Ind_{W(E7)}^{W(E8)}(trivial) ----
fus := PossibleClassFusions(ct7, ct8);;
rperm := fail;;
for f in fus do
  cand := Induced(ct7, ct8, [TrivialCharacter(ct7)], f)[1];
  if cand[1] = 240 and ScalarProduct(ct8, cand, cand) = 5 then rperm := cand; fi;
od;
Print("\n(1) R_perm: degree = ", rperm[1],
      ",  <R_perm,R_perm> = ", ScalarProduct(ct8, rperm, rperm), "\n");
comps := [];;
for i in [1..Length(irr8)] do
  m := ScalarProduct(ct8, rperm, irr8[i]);
  if m > 0 then Add(comps, [irr8[i][1], m]); fi;
od;
Sort(comps);
Print("    decomposition [dim,mult]: ", comps,
      "   sum = ", Sum(comps, c -> c[1]*c[2]), "  (paper: 1+8+35+84+112)\n");

# ---- (2) character of V_112 on the four order-3 classes ----
v112 := irr8[Filtered([1..Length(irr8)], i -> irr8[i][1] = 112)[1]];;
o3 := Filtered([1..Length(ords)], i -> ords[i] = 3);;
Print("\n(2) chi_V112 on order-3 classes: ", List(o3, j -> v112[j]),
      "  (paper: [31,4,4,4])\n");

# ---- (3) fixed roots under the A_2-Coxeter order-3 element ----
a2 := First(o3, j -> cent[j] = 311040);;
Print("\n(3) A2-Coxeter order-3 class ", a2, " (cent 311040, size ", sizes[a2], "):\n");
Print("    R_perm = ", rperm[a2], " roots fixed (paper: 72 = |E6 roots|);  ",
      240 - rperm[a2], " remaining = 3 x ", (240 - rperm[a2])/3, " orbits\n");
QUIT;
