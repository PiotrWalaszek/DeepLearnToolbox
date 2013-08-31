function nnupdatefigures(nn,fhandle,L,opts,i)
%NNUPDATEFIGURES updates figures during training
if i > 1 %dont plot first point, its only a point   
    x_ax = 1:i;
    if opts.validation == 1 
        p = semilogy(x_ax, L.train.e(x_ax), 'b', ...
                 x_ax, L.val.e(x_ax), 'r');
        legend(p, {'Training', 'Validation'},'Location','NorthEast');
    else
        p = semilogy(x_ax,L.train.e(x_ax),'b');
        legend(p, {'Training'},'Location','NorthEast');
    end    
    xlabel('Number of epochs'); ylabel('Error');title('Error');    
    set(gca, 'Xlim',[0,opts.numepochs + 1])

    if i ==1 % speeds up plotting by factor of ~2
        set(gca,'LegendColorbarListeners',[]);
        setappdata(gca,'LegendColorbarManualSpace',1);
        setappdata(gca,'LegendColorbarReclaimSpace',1);

    end
    drawnow;
end
=======
%    plotting
    figure(fhandle);   
    if strcmp(nn.output,'softmax')  %also plot classification error
                
        p1 = subplot(1,2,1);
        plot(plot_x,plot_ye);
        xlabel('Number of epochs'); ylabel('Error');title('Error');
        title('Error')
        legend(p1, M,'Location','NorthEast');
        set(p1, 'Xlim',[0,opts.numepochs + 1])
        
        p2 = subplot(1,2,2);
        plot(plot_x,plot_yfrac);
        xlabel('Number of epochs'); ylabel('Misclassification rate');
        title('Misclassification rate')
        legend(p2, M,'Location','NorthEast');
        set(p2, 'Xlim',[0,opts.numepochs + 1])
        
    else
        
        p = plot(plot_x,plot_ye);
        xlabel('Number of epochs'); ylabel('Error');title('Error');
        legend(p, M,'Location','NorthEast');
        set(gca, 'Xlim',[0,opts.numepochs + 1])
                
    end
    drawnow;
end
end
