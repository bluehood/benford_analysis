#!/usr/bin/python3
import matplotlib.pyplot as plt
print(f'Plot numberline figure for project presentation.')

plt.rcParams.update({'font.size': 13.5})
# set up the figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(0,10)
ax.set_ylim(0,10)

# draw lines
xmin = 1
x_alpha = 3
x_beta = 7
xmax = 9
y = 5
height = 1

#  Main numberline
plt.hlines(y, xmin, xmax, color='blue')
plt.vlines(xmin, y - height / 2., y + height / 2., color='blue')
plt.vlines(xmax, y - height / 2., y + height / 2., color='blue')
plt.vlines(x_alpha, y - height / 4., y + height / 4., color='blue')
plt.vlines(x_beta, y - height / 4., y + height / 4., color='blue')

# Labels numberline
# lambda_c 
plt.hlines(y - 2, xmin, xmax, colors='black')
plt.vlines(xmin, y - 2 -height / 4., y - 2 + height / 4., colors='black')
plt.vlines(xmax, y - 2 -height / 4., y - 2 + height / 4., colors='black')

# lambda_a 
plt.hlines(y + 1, xmin, x_alpha, colors='black')
plt.vlines(xmin, y + 1 -height / 4., y + 1 + height / 4., colors='black')
plt.vlines(x_alpha, y + 1 -height / 4., y + 1 + height / 4., colors='black')

# lambda_b 
plt.hlines(y + 1, x_beta, xmax, colors='black')
plt.vlines(x_beta, y + 1 -height / 4., y + 1 + height / 4., colors='black')
plt.vlines(xmax, y + 1 -height / 4., y + 1 + height / 4., colors='black')

# draw a point on the line
# px = 4
# plt.plot(px,y, 'ro', ms = 15, mfc = 'r')

# # add an arrow
# plt.annotate('Price five days ago', (px,y), xytext = (px - 1, y + 1), 
#               arrowprops=dict(facecolor='black', shrink=0.1), 
#               horizontalalignment='right')

# add numbers
plt.text(xmin + .7, y - 1, r'$a \times 10^{\alpha}$', horizontalalignment='right', color='blue')
plt.text(xmax - .5, y - 1, r'$b \times 10^{\beta}$', horizontalalignment='left', color='blue')
plt.text(x_alpha + .5, y - 1, r'$10^{\alpha + 1}$', horizontalalignment='right', color='blue')
plt.text(x_beta - .2, y - 1, r'$10^{\beta}$', horizontalalignment='left', color='blue')

# lambda labels 
plt.text((xmin + xmax) / 2 , y - 1.75, r'$\lambda_c$', horizontalalignment='right')
plt.text((xmin + x_alpha) / 2 + .15, y + 1.25, r'$\lambda_a$', horizontalalignment='right')
plt.text((x_beta + xmax) / 2 + .15, y + 1.25, r'$\lambda_b$', horizontalalignment='right')
plt.axis('off')
fig.tight_layout()
plt.savefig('/home/odestorm/Documents/physics_project/weekly_reports/project_presentation/figures/finite_range/numberline.png', bbox_inches='tight')
plt.show()