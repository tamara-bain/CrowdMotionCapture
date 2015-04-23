file = ""
cite = 1
with open("Final_LaTex/Report.tex", "r") as ins:
    for line in ins:
        file += line
        if line.startswith('\\bibitem{'):
            line = line.replace('\\bibitem{', '')
            name = line.replace('}\n', '')
            co = '[{0}]'.format(cite)
            cn = '\cite[{0}]'.format(name)
            cn = cn.replace('[', '{')
            cn = cn.replace(']', '}')
            file = file.replace(co, cn)

            cite += 1

print file
