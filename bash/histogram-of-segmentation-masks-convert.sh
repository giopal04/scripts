#!/bin/bash

read fn

if [ -z $fn ] ; then
	fn=$1
fi

output=`convert $fn -define histogram:unique-colors=true -format %c histogram:info:- | awk -F\# '{print $2}'`
echo "$fn "$output

exit 0
