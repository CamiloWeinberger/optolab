function [r,pupil,angle] = find_subpupils2(I_0,tol,rad_fact)
% INPUT:                            Output:
%   -   I_0                             -   r_i coordinates rx1,ry1,rx2,ry2....
%   -   tol Tolarenace                  -   pupil
%   -   raduis factor                   -   angle for rotation

I_0 = double(I_0);I_0 = I_0 - min(I_0(:));I_0 = I_0./max(I_0(:));
pupil = double(I_0>=tol);
max1 = size(I_0,1);
max2 = size(I_0,2);
pup = filter_im(pupil);
[halfs1,halfs2] = cmass(pup);
for kk = 1:4
    if kk == 1
        rr1{kk} = 1:halfs1;
        rr2{kk} = 1:halfs2;
    elseif kk ==2
        rr1{kk} = 1:halfs1;
        rr2{kk} = 1+halfs2:max2;
    elseif kk ==3
        rr1{kk} = 1+halfs1:max1;
        rr2{kk} = 1:halfs2;
    elseif kk ==4
        rr1{kk} = 1+halfs1:max1;
        rr2{kk} = 1+halfs2:max2;
    end
    pup = filter_im(pupil(rr1{kk},rr2{kk}));
    [Y_vec,X_vec]  = find(pup);

    jl = Y_vec*0;
    for idd = 1:length(Y_vec)-2
        jj=Y_vec(idd+2)-Y_vec(idd);
        if jj >=10
            jl(idd) = 1;
        end
    end
    X_vec(jl==1) = [];
    Y_vec(jl==1) = [];
    X1 =  min(X_vec);X2 =  max(X_vec);D(kk) = (X2-X1);
    Y1 =  min(Y_vec);Y2 =  max(Y_vec);D(kk+4) = (Y2-Y1);
end


rad = floor(mean(D)/2)*rad_fact;

for kk = 1:4
    range1 = rr1{kk};
    range2 = rr2{kk};
    [cx,cy]  = cmass(pupil(range1,range2));
    cx = cx+min(range1)-1; cy = cy+min(range2)-1;
    rango = [cx-rad,cx+rad,cy-rad,cy+rad]+.5;
    ran(kk,:) = floor(rango(:));
    CX (kk) = cx;CY(kk)=cy;
end


r1 = [ran(1,1),ran(1,2),ran(1,3),ran(1,4)];
r2 = [ran(2,1),ran(2,2),ran(2,3),ran(2,4)];
r3 = [ran(3,1),ran(3,2),ran(3,3),ran(3,4)];
r4 = [ran(4,1),ran(4,2),ran(4,3),ran(4,4)];

r1 = [r1(1):r1(2);r1(3):r1(4)];
r2 = [r2(1):r2(2);r2(3):r2(4)];
r3 = [r3(1):r3(2);r3(3):r3(4)];
r4 = [r4(1):r4(2);r4(3):r4(4)];
% figure(10);
% subplot(121);imagesc(I_0);axis image; title('RAW image')
% subplot(122);imagesc([I_0(r1(1,:),r1(2,:)),I_0(r2(1,:),r2(2,:));...
% I_0(r3(1,:),r3(2,:)),I_0(r4(1,:),r4(2,:))]); axis image; title(['Cropped image tol = ' num2str(tol)])
pupil = fspecial('disk',rad);
pupil = imresize(pupil,[size(r1,2),size(r1,2)]);

ax = CX-halfs1;
ay = CY-halfs2;
angle = atan(ay./ax)/pi*180;
angle = abs(angle)-45;
angle(1) = -angle(1);
angle(4) = -angle(4);
% display(['rotated ' num2str(angle) ' °'])
angle = mean(angle);

r = [r1;r2;r3;r4];

end


% funcion filtro
function [out] = filter_im(input)
[mm,nn] = size(input);
val = input*0;
for m = 2:mm-1
    for n = 2:nn-1
        val(m,n) = mean(input(m-1:m+1,n-1:n+1),"all");
    end
end
out = (val>.3);
end

function [cx,cy] = cmass(input)
[X_vec,Y_vec]  = find(input);
cx = floor(mean([X_vec]));
cy = floor(mean([Y_vec]));
end