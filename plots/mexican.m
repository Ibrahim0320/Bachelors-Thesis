f = @(r) -1/2*(r.^2) + 1/4*0.05*r.^4;

th = linspace(0,2*pi);
r = linspace(0,5.5);

[R,T] = meshgrid(r,th);

z = f(R);

[x,y] = pol2cart(T,R);
hold on
s = surf(x,y,z, 'FaceAlpha',0.5, 'FaceLighting','gouraud');
s.EdgeAlpha = 0.5;
cmap = colormap(gcf);
colormap(flipud(cmap));
xlabel('Re(\phi)', 'FontSize',14);
ylabel('Im(\phi)','FontSize',14);
zlabel('V(\phi)','FontSize',14);
set(gca,'xtick',[])
set(gca,'ytick',[])
set(gca,'ztick',[]) 

radius = 0.6;
[X,Y,Z] = sphere;
X2 = X*radius;
Y2 = Y*radius;
Z2 = Z*radius+0.3;

X3 = X*radius+4.5;
Y3 = Y*radius;
Z3 = Z*radius-5+0.6;


surf(X2,Y2,Z2);
surf(X3,Y3,Z3);
quiver3(1.5, 0, 0.3, 2.5/2,0,-2,2,'LineWidth',1,'ShowArrowHead','on','Color','r')
