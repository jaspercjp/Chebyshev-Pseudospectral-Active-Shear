import re

def parse(raw_str:str):
    vars_dict = {'Vx':[], 'Vy':[], 'P':[], 'Qxx':[], 'Qxy':[], 'Qxz':[], 'Qyz':[], 'Qzz':[]}
    raw_str = raw_str.replace("\\[Gamma]", 'g').replace("I","j")\
        .replace("\\[Prime]\\[Prime]\\[Prime]\\[Prime]", "D4").replace("\\[Prime]\\[Prime]\\[Prime]", "D3")\
        .replace("\\[Prime]\\[Prime]", "D2").replace("Derivative[1]", "D1")\
        .replace("px","kx").replace("pz","kz").replace("\\[CapitalPi]", "P").replace("p", "k")\
        .replace("q","Q").replace("v","V").replace("\\[Eta]", "eta")
    
    # Decorate the derivatives 
    D_regex = r"[VQ][xyz]*\^D[234]"
    Q_exprs = re.findall(D_regex, raw_str)
    for expr in Q_exprs:
        D = expr.split("^")[1]
        Q = expr.split("^")[0]
        raw_str = raw_str.replace(expr, D+"["+Q+"]")

    # Parse the string based on plus and minus signs outside of parentheses
    regex = "([-\+]\s*(?![^()]*\)))"
    split_str = re.split(regex, raw_str)
    for i in range(len(split_str)):
        token = split_str[i]
        for var in vars_dict.keys():
            if var in token:
                if i>0:
                    symbol = split_str[i-1].strip()
                    vars_dict[var].append(symbol)
                    vars_dict[var].append(token.strip().replace(" ", "*").replace("*+*", "+").replace("*-*", "-"))
                else:
                    # vars_dict[var].append("+")
                    vars_dict[var].append(token.strip().replace(" ", "*").replace("*+*", "+").replace("*-*", "-"))
    return vars_dict

# "Incompressible" refers to using the incompressibility constraint to eliminate a component of velocity
filepath = "../three-dimensions/eqn-derivation-mathematica/3D-equations-incompressible.txt"
outputpath = "../three-dimensions/eqn-derivation-mathematica/translated-3D-equations-incompressible.txt"
with open(filepath) as file:
    with open(outputpath, 'w') as output:
        lines = file.readlines()
        curr_str = ""
        curr_eqn = ""
        for i in range(len(lines)):
            line = lines[i]
            # The equations are expected to be separated by lines of format ![:equation name:]
            # Repeatedly add terms to an accumulator string (curr_str) to parse the current equation
            if not '#' in line and not i==len(lines)-1:
                curr_str += line.replace("\\n",'').strip()
            else:
                if i>0:
                    vars_dict = parse(curr_str)
                    for key in vars_dict.keys():
                        # Remove the variable names from the differentiation matrices
                        var_str = str(" ".join(vars_dict[key])).replace("^", "**")
                        regex = r"\[[VQ][xyz]*\]"
                        exprs = re.findall(regex, var_str)
                        for expr in exprs:
                            var_str = var_str.replace(expr, "")
                        regex = r"\[P\]"
                        exprs = re.findall(regex, var_str)
                        for expr in exprs:
                            var_str = var_str.replace(expr, "")

                        # Replace the variable names with the identity matrix
                        regex = r"[VQ][xyz]*"
                        exprs = re.findall(regex, var_str)
                        for expr in exprs:
                            var_str = var_str.replace(expr, "II")
                        regex = r"P"
                        exprs = re.findall(regex, var_str)
                        for expr in exprs:
                            var_str = var_str.replace(expr, "II")

                        # Write the result string to output
                        curr_eqn_name = curr_eqn.split(' ')[0][1:]
                    
                        if not var_str == "":
                            if var_str[0] == '+':
                                var_str = var_str[2:] # erase leading plus signs
                            if "Q" in key and curr_eqn_name==key:
                                output.write(f"LHS[R{curr_eqn_name}, R{key}] = SPEC_OP_2\n")
                            else:
                                output.write(f"LHS[R{curr_eqn_name}, R{key}] = "+ var_str +"\n")
                if "#" in line:
                    output.write("\n"+line)
                    curr_eqn = line
                curr_str = ""   
print("Wrote output to" + outputpath)