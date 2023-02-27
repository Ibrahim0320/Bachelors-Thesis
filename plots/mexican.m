f = @(x) -1/2*(x.^2) + 1/4*0.05*x.^4;

lim=8

x = linspace(-lim,lim);
hold on
plot(x, f(x), 'r', LineWidth=1)
grid on
set(gca, 'xtick', [])
set(gca, 'ytick', [])
xlabel('\phi')
ylabel('V(\phi)')