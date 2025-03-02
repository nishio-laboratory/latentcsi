#!/bin/sh

if [ -z "$(docker images | grep csi_image)" ]
then
    docker build -t csi_image .
fi

while getopts 'bpr' arg; do
    case $arg in
        p)
            docker run --ipc=host -v /mnt/nas/esrh/csi_image_data:/data -v .:/code --gpus="device=all" -it csi_image ipython -i --simple-prompt --InteractiveShell.display_page=True
            ;;
        b)
            docker run --ipc=host -v /mnt/nas/esrh/csi_image_data:/data -v .:/code --gpus="device=all" -it csi_image bash
            ;;
        r)
            docker ps | grep "csi_image" | cut -d " " -f 1 | xargs docker rm -f
            ;;
    esac
done
