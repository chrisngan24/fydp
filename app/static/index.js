
var headChartID = "headChart";
var wheelChartID = "wheelChart";
var vPlayer = null;

var video = videojs("annotated_vid", {
  controls: true,
    autoplay: false,
    preload: 'auto',
    plugins: {
      framebyframe: {
        fps: videoData['fps'],
        steps: [
          { text: '-5', step: -5 },
          { text: '-2', step: -2 },
          { text: '+2', step: 2 },
          { text: '+5', step: 5 },
        ]
      }
    }
});
var videoID = video.id() + '_html5_api';


var headData = [
  {
    label: 'Head',
    strokeColor: '#F16220',
    pointColor: '#F16220',
    pointStrokeColor: '#fff',
    data: fusedData.headData,
  },
];


var wheelData = [
  {
    label: 'Wheel',
    strokeColor: '#F16220',
    pointColor: '#F16220',
    pointStrokeColor: '#fff',
    data: fusedData.wheelData,
  },
];

var width = 20;
var chartOptions = {
  tooltipTemplate: "<%if (datasetLabel){%><%=datasetLabel%>: <%}%><%=argLabel%>, <%=valueLabel%>",
  xScaleOverride : true,
  xScaleSteps : Math.round(videoData['frames']/width),
  xScaleStepWidth:width,
  xScaleStartValue: 0,
  pointDotRadius:3, 

};

var headCtx = document.getElementById(headChartID).getContext("2d");
var headChart = new Chart(headCtx).VideoChart(headData, chartOptions);
headChart.setSentimentEvents(fusedData['headEvents']);

var wheelCtx = document.getElementById(wheelChartID).getContext("2d");
var wheelChart = new Chart(wheelCtx).VideoChart(wheelData,chartOptions);
wheelChart.setSentimentEvents(fusedData['laneEvents']);

function updateCharts() {
  var canvasWidth = $('#' + headChartID).width();
  var x = video.currentTime()/videoData['video_time'] * (canvasWidth - CHART_PADDING) + CHART_PADDING;
  headChart.updateLine(x);
  wheelChart.updateLine(x);
  console.log('OOGAF');
}
updateCharts();
var onVideoClick = function(e){
  var totalOffsetX = 0;
  var canvasX = 0;
  var currentElement = this;

  do{
    totalOffsetX += currentElement.offsetLeft - currentElement.scrollLeft;
  }
  while(currentElement = currentElement.offsetParent)

  canvasX = event.pageX - totalOffsetX;
  var x = canvasX;
  if ( x > CHART_PADDING){
    headChart.updateLine(x);
    wheelChart.updateLine(x);
    console.log(canvasX);
    var vidTime = (canvasX - CHART_PADDING)/(this.clientWidth - CHART_PADDING)*videoData['video_time'];
    video.currentTime(vidTime);
  }
}
$('#' + headChartID).on('click', onVideoClick);
$('#' + wheelChartID).on('click', onVideoClick);


var repeater;


$('#' + videoID).on('play', function(){
  function doWork() {
    updateCharts();
    repeater = setTimeout(doWork, 100);
  }
  doWork();
});

$('#' + videoID).on('pause', function(){
  console.log('stop');
  updateCharts();
  clearTimeout(repeater);
})


$('#' + videoID).on('seeking', function(){
  // events including moving video time and skipping frames
  updateCharts();
});
