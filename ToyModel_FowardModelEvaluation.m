

close all

% Function
linear_factor = 10;
syms g(x)
g(x) = ( (x+linear_factor).^2 - linear_factor^2 ) / ((1+linear_factor).^2 - linear_factor^2);

% linearization
mu_x = 0.5;
G = ( 2*( mu_x + linear_factor ) ) / ((1+linear_factor).^2 - linear_factor^2)
G_const = g(mu_x) - G*mu_x

x_axis = linspace(0,1,100);
figure
plot(x_axis, double(g(x_axis)) )
hold all
plot(x_axis, G*x_axis + G_const )