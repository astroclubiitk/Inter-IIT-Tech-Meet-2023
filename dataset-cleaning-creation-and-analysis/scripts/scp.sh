for item in *.img do
	lronac2isis from=$1 to=$1.raw.cub;
	spiceinit from=$1.raw.cub spksmithed=true web=true;
	lronaccal from=$1.raw.cub to=$1.cal.cub
	lronacecho from=$1.cal.cub to=$1.echo.cub
	isis2std from=$1.echo.cub to=$1.png
done