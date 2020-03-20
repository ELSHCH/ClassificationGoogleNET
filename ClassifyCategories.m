function yLabel=ClassifyCategories(y,x_class,x_val);
dimX=length(x_class);
lengthNw=length(y(:,1,1));
lengthNd=length(y(1,:,1));
for i=1:dimX-1
        for k1=1:lengthNw
            for k2=1:lengthNd
              [c,ia]=intersect(y(k1,k2,:),max(y(k1,k2,:)));
              x_max=x_val(ia);

             % plot(1:length(y(k1,k2,:)),y(k1,k2,:));
        if (x_max>=x_class(i))&&(x_max<=x_class(i+1))
            k(k1,k2)=i;   
        end;
            end;
        end;
end;  
yLabel= cell(lengthNw,1);
for k1=1:lengthNw
    world1='';
            for k2=1:lengthNd       
            world1=strcat(world1,num2str(k(k1,k2)));
        end;
        yLabel(k1)=cellstr(world1);
            end;       
%plot(1:lengthNw,yd(:)) ;  