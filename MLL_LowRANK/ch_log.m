function [y_log] = ch_log(Y,t)
[num_instances,num_label]=size(Y);
y_log=zeros(num_instances,num_label);
[y_label,index]=sort(Y,2,'descend');
    for i=1:num_instances
        s=0;
        for j=1:num_label
            if s<=t
                s=s+y_label(i,j);
                y_log(i,index(i,j))=1;
            end
        end
    end
end
    


