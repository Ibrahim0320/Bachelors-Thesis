f = @(r) -1/2*(r.^2) + 1/4*0.05*r.^4;

th = linspace(0,2*pi);
r = linspace(0,5.5);

[R,T] = meshgrid(r,th);

z = f(R);

[x,y] = pol2cart(T,R);

s = surf(x,y,z, 'FaceAlpha',0.5, 'FaceLighting','gouraud');
s.EdgeAlpha = 0.5;

xlabel('Re(\phi)', 'FontSize',14);
ylabel('Im(\phi)','FontSize',14);
zlabel('V(\phi)','FontSize',14);
set(gca,'xtick',[])
set(gca,'ytick',[])
set(gca,'ztick',[]) 