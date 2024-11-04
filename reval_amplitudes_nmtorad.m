



r0_init = .01:.01:.2;
%r0_init = .01:.01:.06;
%r0_init = .07:.01:.2;
parts = 1
resol = 268;
V=5;
parts = 1;
for D = [1.5]
    carpeta_1  = ['Dataset_Phasemap/Datasets_phasemap_D' num2str(D) '_Nov_2022_test/']; %phasemap
    carpeta_1  = ['Dataset_Phasemap/Datasets_phasemap_D' num2str(D) '_Apr_2023/']; %phasemap

    % arpeta = ['Dataset_pyramid/Datasets_phase2pyr_D' num2str(D) '_Nov_2022_M' num2str(Mod) '_ron' num2str(ron) '_photonN' num2str(photonNoise) '_test']; % to save
    for part = 1:parts
        part
        for r0 = r0_init
            r0
            name_in = ['./' carpeta_1 '/Phase_' num2str(resol) 'px_r0_' num2str(r0) '_part_' num2str(part) '.mat'];
            %name_out = ['./' carpeta '/Raw_' num2str(resol) 'px_r0_' num2str(r0) '_part_' num2str(part) '.mat'];
            load(name_in)
            %Piston for BLT
            Y_z = Y_z*2*pi/550; % set in rads
            %[X_s,~] = Pyr2raw(X_phase,size(X_phase,3),sys);
            save(name_in,'X_phase','Y_z','r0_var','wvl_factor');
        end
    end
end