classdef Reg_CMAA_TTV < handle  % ig.mask, sizeD, mu, M1, M2, M3, U_x,U_y,U_z,V_x,V_y,V_z
  
  properties
    mMask;    % the mask matrix
    sizeD;
    mu;   % ADMM parameter
    M1;   % 
    M2;  % 
    M3;
    U_x;    % 
    U_y;   % 
    U_z;     % 
    V_x;   %
    V_y;
    V_z;     
  end
  
  methods
    function obj = Reg_CMAA_TTV(mask, sizeD, mu, M1, M2, M3, U_x,U_y,U_z,V_x,V_y,V_z)
      obj.mMask = mask;
      obj.sizeD=sizeD;
      obj.mu = mu;
      obj.M1 = M1;
      obj.M2 = M2;
      obj.M3 = M3;
      obj.U_x = U_x;
      obj.U_y = U_y;
      obj.U_z = U_z;
      obj.V_x = V_x;      
      obj.V_y = V_y;
      obj.V_z = V_z; 
            
    end
    
   
    function grad = cgrad(obj, xD)
      xD_full=zeros(obj.sizeD(1)*obj.sizeD(2),obj.sizeD(3));  
      for ii=1:obj.sizeD(3)
          tmp=embed(xD(:,ii),obj.mMask);          
          xD_full(:,ii)=tmp(:);
      end       
      
      % Dx
      tmp_x = diff_x(xD_full,obj.sizeD);
      x_err=tmp_x-col(obj.U_x*obj.V_x')+col(obj.M1/obj.mu(1));
      Dx_err=obj.mu(1)*diff_xT(x_err,obj.sizeD);
      % Dy
      tmp_y = diff_y(xD_full,obj.sizeD);
      y_err=tmp_y-col(obj.U_y*obj.V_y')+col(obj.M2/obj.mu(2));
      Dy_err=obj.mu(2)*diff_yT(y_err,obj.sizeD);      
      % Dz
      tmp_z = diff_z(xD_full,obj.sizeD);
      z_err=tmp_z-col(obj.U_z*obj.V_z')+col(obj.M3/obj.mu(3));
      Dz_err=obj.mu(3)*diff_zT(z_err,obj.sizeD);  
      
      grad_full=Dx_err+Dy_err+Dz_err;
      grad_full_reshape=reshape(grad_full,[obj.sizeD(1)*obj.sizeD(2),obj.sizeD(3)]);
      grad=[];
      for ii=1:obj.sizeD(3)
          tmp = grad_full_reshape(:,ii);
          tmp_mask=tmp(obj.mMask);
          grad=[grad;tmp_mask];
      end
    end
    
    
   
  end
  
end

