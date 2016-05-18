function HON4D_OfCell = GetHON4D_OfCell(dx_img,dy_img,dz_img,cell,P)
DIM = cell.DIM;
xStep = cell.width/cell.numC; % 14
yStep = cell.height/cell.numR;% 14

HON4D_OfCell = [];

    % 对cell 中的每个 grid 计算 HON4D 再按行联结起来
    for cj=1:cell.numC
        for ci=1:cell.numR
            beginX = xStep*(cj-1)+1;
            beginY = yStep*(ci-1)+1;
            
            xmin = beginX;
            ymin = beginY;
            xmax = xmin + xStep -1;
            ymax = ymin + yStep -1;
            
            %越界防护操作, 已知多余
            %ymax = max(min(ymax,size(dx_img,1)),1);
            %xmax = max(min(xmax,size(dx_img,2)),1);
            %ymin = max(ymin,1);
            %xmin = max(xmin,1);
            
            gx_c = dx_img([ymin:ymax],[xmin:xmax],:);
            gy_c = dy_img([ymin:ymax],[xmin:xmax],:);
            gz_c = dz_img([ymin:ymax],[xmin:xmax],:);
            
            gx_c = gx_c(:);
            gy_c = gy_c(:);
            gz_c = gz_c(:);
            
            validInd = intersect(intersect(find(gx_c~=0),find(gy_c~=0)),find(gz_c~=0));
            
            gx_c = gx_c(validInd);
            gy_c = gy_c(validInd);
            gz_c = gz_c(validInd);
            
            
            
            gmags = sqrt(gx_c.^2 + gy_c.^2 + gz_c.^2);
            gmags_rep = repmat(gmags,1,DIM);
            
            gmat = [gx_c,gy_c,gz_c,-1*ones(length(gx_c),1)];
               
            res = (P*gmat')';
            
            res_norm = res./gmags_rep;
            
            res_norm(isinf(res_norm)) = 0;
            res_norm(isnan(res_norm)) = 0;
            
            
            res_norm = res_norm - 1.3090;
            res_norm(res_norm<0) = 0;
            
            
            vecmag = sqrt(sum(res_norm.^2,2));
            vecmag_rep = repmat(vecmag,1,DIM);
            
            
            res_norm = res_norm./vecmag_rep;
            res_norm(isinf(res_norm)) = 0;
            res_norm(isnan(res_norm)) = 0;
            
            
            res_norm = res_norm./length(validInd);
              
            HON4D_OfCell = [HON4D_OfCell,sum(res_norm,1)];
        end
    end
    
end