clear all
close all
clc

paths{1} = './Dataset_Phasemap/Datasets_phasemap_D1.5_Nov_2022/'
paths{2} = './Dataset_Phasemap/Datasets_phasemap_D1.5_Apr_2023/'

%figure(1);hold off

Yz = [];
for idx = 1:2
    path_f = dir(paths{idx});
    fold = paths{idx}
    Y = [];
    for idz = 1:length(path_f)-2
        Y_z = load([fold path_f(idz+2).name]).Y_z;
        Y = cat(1,Y, Y_z(:,1:20));
    end
    Yz{idx} = Y;
end

save('Zn20_vals','Yz','paths');

