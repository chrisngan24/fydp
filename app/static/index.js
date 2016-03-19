
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
}).ready(function () {
  vPlayer = this;
});


var data = [
  {
    label: 'My First dataset',
    strokeColor: '#F16220',
    pointColor: '#F16220',
    pointStrokeColor: '#fff',
    data: [
      { x: 19, y: 65 }, 
      { x: 27, y: 59 }, 
      { x: 28, y: 69 }, 
      { x: 40, y: 81 },
      { x: 48, y: 56 }
    ]
  },
  {
    label: 'My Second dataset',
    strokeColor: '#007ACC',
    pointColor: '#007ACC',
    pointStrokeColor: '#fff',
    data: [
      { x: 19, y: 75  }, 
      { x: 27, y: 69  }, 
      { x: 28, y: 70  }, 
      { x: 40, y: 31  },
      { x: 48, y: 76  },
      { x: 52, y: 23  }, 
      { x: 24, y: 32  }
    ]
  }
];
var headCtx = document.getElementById(headChartID).getContext("2d");
var headChart = new Chart(headCtx).VideoChart(data);

var wheelCtx = document.getElementById(wheelChartID).getContext("2d");
var wheelChart = new Chart(wheelCtx).VideoChart(data);

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
  headChart.updateLine(x);
  wheelChart.updateLine(x);

}
$('#' + headChartID).on('click', onVideoClick);
$('#' + wheelChartID).on('click', onVideoClick);

