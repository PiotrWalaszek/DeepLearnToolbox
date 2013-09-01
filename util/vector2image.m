function image_out = vector2image(X)
    s = sqrt(size(X,2));
    im = zeros(s,s);
    for i = 1:s
       im(i,:) = X(1,((i-1)*s +1):(i*s)); 
    end
    
    if nargout==1
        image_out = im;
    else
        imagesc(im, [min(X(1,:)) max(X(1,:))]);
        axis equal
        colormap gray
    end
end
